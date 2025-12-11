import datetime
import warnings
from pathlib import Path

import click
import pandas as pd
import gc
import cv2
import torch
from tifffile import imread, imwrite
from tqdm import tqdm

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def refresh_cuda_memory():
    """
    Re-allocate all cuda memory to help alleviate fragmentation
    """
    # Run a full garbage collect first so any dangling tensors are released
    gc.collect()

    # Then move all tensors to the CPU
    locations = {}
    for obj in gc.get_objects():
        if not isinstance(obj, torch.Tensor):
            continue

        locations[obj] = obj.device
        obj.data = obj.data.cpu()
        if isinstance(obj, torch.nn.Parameter) and obj.grad is not None:
            obj.grad.data = obj.grad.cpu()

    # Now empty the cache to flush the allocator
    torch.cuda.empty_cache()

    # Finally move the tensors back to their associated GPUs
    for tensor, device in locations.items():
        tensor.data = tensor.to(device)
        if isinstance(tensor, torch.nn.Parameter) and tensor.grad is not None:
            tensor.grad.data = tensor.grad.to(device)


class TI_Loss(torch.nn.Module):
    def __init__(self, dim, connectivity, inclusion, exclusion, min_thick=1):
        """
        :param dim: 2 if 2D; 3 if 3D
        :param connectivity: 4 or 8 for 2D; 6 or 26 for 3D
        :param inclusion: list of [A,B] classes where A is completely surrounded by B.
        :param exclusion: list of [A,C] classes where A and C exclude each other.
        :param min_thick: Minimum thickness/separation between the two classes. Only used if connectivity is 8 for 2D or 26 for 3D
        """
        super(TI_Loss, self).__init__()

        self.dim = dim
        self.connectivity = connectivity
        self.min_thick = min_thick
        self.interaction_list = []
        self.sum_dim_list = None
        self.conv_op = None
        self.apply_nonlin = lambda x: torch.nn.functional.softmax(x, 1)
        self.ce_loss_func = torch.nn.CrossEntropyLoss(reduction="none")

        if self.dim == 2:
            self.sum_dim_list = [1, 2, 3]
            self.conv_op = torch.nn.functional.conv2d
        elif self.dim == 3:
            self.sum_dim_list = [1, 2, 3, 4]
            self.conv_op = torch.nn.functional.conv3d

        self.set_kernel()

        for inc in inclusion:
            temp_pair = []
            temp_pair.append(True)  # type inclusion
            temp_pair.append(inc[0])
            temp_pair.append(inc[1])
            self.interaction_list.append(temp_pair)

        for exc in exclusion:
            temp_pair = []
            temp_pair.append(False)  # type exclusion
            temp_pair.append(exc[0])
            temp_pair.append(exc[1])
            self.interaction_list.append(temp_pair)

    def set_kernel(self):
        """
        Sets the connectivity kernel based on user's sepcification of dim, connectivity, min_thick
        """
        k = 2 * self.min_thick + 1
        if self.dim == 2:
            if self.connectivity == 4:
                np_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            elif self.connectivity == 8:
                np_kernel = np.ones((k, k))

        elif self.dim == 3:
            if self.connectivity == 6:
                np_kernel = np.array(
                    [
                        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    ]
                )
            elif self.connectivity == 26:
                np_kernel = np.ones((k, k, k))

        self.kernel = torch_kernel = torch.from_numpy(
            np.expand_dims(np.expand_dims(np_kernel, axis=0), axis=0)
        )
        self.kernel = self.kernel.cuda()

    def topological_interaction_module(self, P, mode="default"):
        """
        Given a discrete segmentation map and the intended topological interactions, this module computes the critical voxels map.
        :param P: Discrete segmentation map
        :return: Critical voxels map
        """

        if mode == "default":
            kernel = np.ones((3, 3, 3))
            kernel = torch.from_numpy(
                np.expand_dims(np.expand_dims(kernel, axis=0), axis=0)
            )
            kernel = kernel.cuda()

        elif mode == "slice":
            kernel = np.array(
                [
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                ]
            )
            kernel = torch.from_numpy(
                np.expand_dims(np.expand_dims(kernel, axis=0), axis=0)
            )
            kernel = kernel.cuda()

        else:
            raise ValueError("Invalid mode.")

        for ind, interaction in enumerate(self.interaction_list):
            interaction_type = interaction[0]
            label_A = interaction[1]
            label_C = interaction[2]

            # Get Masks
            mask_A = torch.where(P == label_A, 1.0, 0.0).double()
            if interaction_type:
                mask_C = torch.where(P == label_C, 1.0, 0.0).double()
                mask_C = torch.logical_or(mask_C, mask_A).double()
                mask_C = torch.logical_not(mask_C).double()
            else:
                mask_C = torch.where(P == label_C, 1.0, 0.0).double()

            # Get Neighbourhood Information
            neighbourhood_C = self.conv_op(mask_C, kernel.double(), padding="same")
            neighbourhood_C = torch.where(neighbourhood_C >= 1.0, 1.0, 0.0)
            neighbourhood_A = self.conv_op(mask_A, kernel.double(), padding="same")
            neighbourhood_A = torch.where(neighbourhood_A >= 1.0, 1.0, 0.0)

            # Get the pixels which induce errors
            violating_A = neighbourhood_C * mask_A
            violating_C = neighbourhood_A * mask_C
            violating = violating_A + violating_C
            violating = torch.where(violating >= 1.0, 1.0, 0.0)

            if ind == 0:
                critical_voxels_map = violating
            else:
                critical_voxels_map = torch.logical_or(
                    critical_voxels_map, violating
                ).double()

        return critical_voxels_map

    def forward(self, x, y):
        """
        The forward function computes the TI loss value.
        :param x: Likelihood map of shape: b, c, x, y(, z) with c = total number of classes
        :param y: GT of shape: b, c, x, y(, z) with c=1. The GT should only contain values in [0,L) range where L is the total number of classes.
        :return:  TI loss value
        """

        if x.device.type == "cuda":
            self.kernel = self.kernel.cuda(x.device.index)

        # Obtain discrete segmentation map
        x_softmax = self.apply_nonlin(x)
        P = torch.argmax(x_softmax, dim=1)
        P = torch.unsqueeze(P.double(), dim=1)
        del x_softmax

        # Call the Topological Interaction Module
        critical_voxels_map = self.topological_interaction_module(P)

        # Compute the TI loss value
        ce_tensor = torch.unsqueeze(
            self.ce_loss_func(x.double(), y[:, 0].long()), dim=1
        )
        ce_tensor[:, 0] = ce_tensor[:, 0] * torch.squeeze(critical_voxels_map, dim=1)
        ce_loss_value = ce_tensor.sum(dim=self.sum_dim_list).mean()

        return ce_loss_value

def contour_3d(y):
    n_classes = np.unique(y)
    cnt_map = np.zeros_like(y)
    for i in range(1, len(n_classes)):
        for j in range(y.shape[0]):
            cnt_tmp, _ = cv2.findContours(
                np.uint8(y[j] == n_classes[i]),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(cnt_map[j], cnt_tmp, -1, i, 2)
    return cnt_map


def erode_3d(y, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = np.zeros_like(y)
    for i in range(y.shape[0]):
        eroded[i] = cv2.erode(y[i].astype(np.uint8), kernel, iterations=1)
    return eroded


def sanity_check(model):
    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    assert data_dir.exists(), f"Data directory not found: {data_dir}"

    gt_path = data_dir / "gt"
    if model == "gt":
        pred_path = gt_path
    else:
        pred_path = data_dir / model / "pred"

    gt_files = sorted(gt_path.glob("*.tif"))
    pred_files = sorted(pred_path.glob("*.tif"))

    assert gt_files, f"No ground truth files found in {gt_path}"
    assert pred_files, f"No prediction files found in {pred_path}"
    assert len(gt_files) == len(pred_files), "Number of files do not match"
    for gt_file, pred_file in zip(gt_files, pred_files):
        assert gt_file.stem == pred_file.stem, "File names do not match"


def calculate_metric(model):
    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    res_dir = Path(__file__).parent.parent / "results" / model
    res_dir.mkdir(parents=True, exist_ok=True)

    gt_path = data_dir / "gt"
    if model == "gt":
        pred_path = gt_path
    else:
        pred_path = data_dir / model / "pred"

    gt_files = sorted(gt_path.glob("*.tif"))
    pred_files = sorted(pred_path.glob("*.tif"))

    # create a directory to store critical voxels map
    critical_voxels_dir = res_dir / "critical_voxels"
    critical_voxels_dir.mkdir(parents=True, exist_ok=True)

    # metrics storage
    topo_error = []

    for gt_file, pred_file in tqdm(zip(gt_files, pred_files), total=len(gt_files)):
        gt_arr = imread(gt_file)
        pred_arr = imread(pred_file)
        num_pixel = np.unique(pred_arr, return_counts=True)
        pred_arr = np.expand_dims(pred_arr, axis=0)

        pred = torch.tensor(pred_arr).cuda()

        general_error = TI_Loss(
            dim=3,
            connectivity=26,
            inclusion=[[1, 2]],
            exclusion=[[0, 1]],
            min_thick=1,
        )
        general_error = general_error.cuda()

        slice_error = TI_Loss(
            dim=3,
            connectivity=26,
            inclusion=[],
            exclusion=[[0, 2], [0, 1], [1, 2]],
            min_thick=1,
        )
        slice_error = slice_error.cuda()

        general_error_map = general_error.topological_interaction_module(
            pred, mode="default"
        )
        slice_error_map = slice_error.topological_interaction_module(pred, mode="slice")

        general_error_voxels = torch.sum(general_error_map)

        general_error_map = general_error_map.squeeze().cpu().numpy()
        slice_error_map = slice_error_map.squeeze().cpu().numpy()

        kernel = np.ones((3, 3), np.uint8)
        new_slice_error_map = erode_3d(slice_error_map, kernel_size=3)

        total_error_map = np.logical_or(general_error_map, new_slice_error_map)

        imwrite(
            critical_voxels_dir / f"{pred_file.stem}_general_error.tif",
            general_error_map.astype(np.uint8) * 255,
            compression="lzw",
        )
        imwrite(
            critical_voxels_dir / f"{pred_file.stem}_slice_error_no_contour.tif",
            new_slice_error_map.astype(np.uint8) * 255,
            compression="lzw",
        )

        topo_error.append(
            [
                pred_file.stem,
                num_pixel[1][0],  # bg total pixel
                num_pixel[1][1],  # fas total pixel
                num_pixel[1][2],  # epi total pixel
                general_error_voxels.item(),
                np.count_nonzero(new_slice_error_map),
                np.count_nonzero(total_error_map),
            ]
        )

        refresh_cuda_memory()

    timestamp = datetime.datetime.now().strftime("%Y%m%d")

    # Save results
    if topo_error:
        df = pd.DataFrame(
            topo_error,
            columns=[
                "id",
                "bg_total_pixel",
                "fas_total_pixel",
                "epi_total_pixel",
                "general_error_voxels",
                "slice_error_voxels",
                "total_error_voxels",
            ],
        )
        df.to_csv(res_dir / f"TEST_topo_error_{timestamp}.csv", index=False)


@click.command()
@click.option("--model", help="Model name", multiple=True, required=True)
def main(model):
    model_ls = list(model)

    for m in model_ls:
        sanity_check(m)

    for m in model_ls:
        print(f"Calculating metrics for {m}")
        calculate_metric(m)


if __name__ == "__main__":
    main()
