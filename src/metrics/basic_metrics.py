import datetime
import warnings
from pathlib import Path

import click
import pandas as pd
import torch
from monai.metrics import (
    compute_average_surface_distance,
    compute_dice,
    compute_hausdorff_distance,
    compute_surface_dice,
    get_confusion_matrix,
)
from monai.transforms import AsDiscrete
from tifffile import imread
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3
SUPPORTED_METRICS = (
    "dice",
    "hausdorff",
    "confusion",
    "surface_dice",
    "frame_surface_dice",
    "asd",
)
DEFAULT_METRICS = SUPPORTED_METRICS
SURFACE_DICE_CLASS_THRESHOLDS = (1, 1)
FRAME_SURFACE_DICE_THRESHOLDS = (0, 0)


def to_onehot(
    x, num_classes=NUM_CLASSES, add_channel_dim=False, add_batch_dim=False
):
    if add_channel_dim:
        x = x.unsqueeze(0)
    x = AsDiscrete(to_onehot=num_classes)(x)
    if add_batch_dim:
        x = x.unsqueeze(0)
    return x


def dice_tensor(gt, pred):
    """
    Compute the Dice score between two binary images using GPU.
    """
    sample_dice = compute_dice(pred, gt, include_background=False)
    sample_dice = sample_dice.squeeze(0)
    fas_dice = sample_dice[0].item()
    epi_dice = sample_dice[1].item()

    return fas_dice, epi_dice


def hausdorff_tensor(gt, pred):
    """
    Compute the Hausdorff score between two binary images using GPU.
    """
    hd = compute_hausdorff_distance(pred, gt, percentile=50, include_background=False)
    hd = hd.squeeze(0)
    fas_hd = hd[0].item()
    epi_hd = hd[1].item()

    return fas_hd, epi_hd


def surface_dice_tensor(gt, pred, cls_th=None):
    """
    Compute the surface Dice score between two binary images.
    """
    if cls_th is None:
        cls_th = SURFACE_DICE_CLASS_THRESHOLDS
    asd = compute_surface_dice(
        pred,
        gt,
        include_background=False,
        class_thresholds=cls_th,
    )
    asd = asd.squeeze(0).detach().cpu().numpy()
    fas_asd = asd[0]
    epi_asd = asd[1]

    return fas_asd, epi_asd


def frame_surface_dice_tensor(pred, cls_th=None):
    """
    Compute the frame surface Dice score between consecutive slices.
    """
    if cls_th is None:
        cls_th = FRAME_SURFACE_DICE_THRESHOLDS
    num_slice = pred.shape[2]
    res_ls = []
    for idx in range(1, num_slice):
        prev_slice = pred[:, :, idx - 1, :, :].detach().clone()
        current_slice = pred[:, :, idx, :, :].detach().clone()
        with torch.no_grad():
            asd = compute_surface_dice(
                prev_slice,
                current_slice,
                include_background=False,
                class_thresholds=cls_th,
            )
        asd = asd.squeeze(0).detach().cpu().numpy()
        fas_asd = asd[0]
        epi_asd = asd[1]
        res_ls.append([idx - 1, idx, fas_asd, epi_asd])
        del asd

        del prev_slice, current_slice
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return res_ls


def asd_tensor(gt, pred):
    """
    Compute the Average Surface Distance between two binary images using GPU.
    """
    asd = compute_average_surface_distance(
        pred, gt, include_background=False, symmetric=True
    )
    asd = asd.squeeze(0)
    fas_asd = asd[0].item()
    epi_asd = asd[1].item()

    return fas_asd, epi_asd


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


def calculate_metric(model, metrics):
    data_dir = Path(__file__).parent.parent / "data"
    res_dir = Path(__file__).parent.parent / "results" / model
    res_dir.mkdir(parents=True, exist_ok=True)

    gt_path = data_dir / "gt"
    pred_path = gt_path if model == "gt" else data_dir / model / "pred"

    gt_files = sorted(gt_path.glob("*.tif"))
    pred_files = sorted(pred_path.glob("*.tif"))

    requested = set(metrics)
    results = {name: [] for name in requested}

    for gt_file, pred_file in tqdm(zip(gt_files, pred_files), total=len(gt_files)):
        gt_arr = imread(gt_file)
        pred_arr = imread(pred_file)

        assert gt_arr.shape == pred_arr.shape, "Shape mismatch"

        gt = torch.as_tensor(gt_arr, device=DEVICE)
        pred = torch.as_tensor(pred_arr, device=DEVICE)

        gt = to_onehot(gt, add_channel_dim=True, add_batch_dim=True)
        pred = to_onehot(pred, add_channel_dim=True, add_batch_dim=True)

        if "dice" in requested:
            fas_dice, epi_dice = dice_tensor(gt, pred)
            results["dice"].append([gt_file.stem, fas_dice, epi_dice])

        if "confusion" in requested:
            conf_m = get_confusion_matrix(pred, gt, include_background=True)
            conf_m = conf_m.squeeze(0).cpu().numpy()
            results["confusion"].append([gt_file.stem, conf_m])

        if "hausdorff" in requested:
            fas_hd, epi_hd = hausdorff_tensor(gt, pred)
            results["hausdorff"].append([gt_file.stem, fas_hd, epi_hd])

        if "surface_dice" in requested:
            fas_surface_dice, epi_surface_dice = surface_dice_tensor(
                gt, pred, cls_th=SURFACE_DICE_CLASS_THRESHOLDS
            )
            results["surface_dice"].append(
                [gt_file.stem, fas_surface_dice, epi_surface_dice]
            )

        if "frame_surface_dice" in requested:
            frame_surface_dice = frame_surface_dice_tensor(
                pred, cls_th=FRAME_SURFACE_DICE_THRESHOLDS
            )
            results["frame_surface_dice"].extend(
                [[gt_file.stem, *x] for x in frame_surface_dice]
            )

        if "asd" in requested:
            fas_asd, epi_asd = asd_tensor(gt, pred)
            results["asd"].append([gt_file.stem, fas_asd, epi_asd])

        del gt, pred, gt_arr, pred_arr
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    timestamp = datetime.datetime.now().strftime("%Y%m%d")

    dice_scores = results.get("dice")
    if dice_scores:
        dice_scores_ls = []
        for img_id, fas_dice, epi_dice in dice_scores:
            dice_scores_ls.append([img_id, "fas", fas_dice])
            dice_scores_ls.append([img_id, "epi", epi_dice])
        dice_df = pd.DataFrame(dice_scores_ls, columns=["id", "cls", "dice"])
        dice_df.to_csv(res_dir / f"dice_scores_{timestamp}.csv", index=False)

    hausdorff_scores = results.get("hausdorff")
    if hausdorff_scores:
        hausdorff_scores_ls = []
        for img_id, fas_hd, epi_hd in hausdorff_scores:
            hausdorff_scores_ls.append([img_id, "fas", fas_hd])
            hausdorff_scores_ls.append([img_id, "epi", epi_hd])
        hausdorff_df = pd.DataFrame(hausdorff_scores_ls, columns=["id", "cls", "hd95"])
        hausdorff_df.to_csv(res_dir / f"hausdorff_scores_{timestamp}.csv", index=False)

    conf_matrix = results.get("confusion")
    if conf_matrix:
        conf_matrix_ls = []
        for img_id, matrix in conf_matrix:
            bg_m, fas_m, epi_m = matrix
            conf_matrix_ls.append([img_id, "bg", *bg_m])
            conf_matrix_ls.append([img_id, "fas", *fas_m])
            conf_matrix_ls.append([img_id, "epi", *epi_m])

        conf_df = pd.DataFrame(
            conf_matrix_ls, columns=["id", "class", "tp", "fp", "tn", "fn"]
        )
        conf_df.to_csv(res_dir / f"confusion_matrix_{timestamp}.csv", index=False)

    surface_dice_scores = results.get("surface_dice")
    if surface_dice_scores:
        surface_dice_scores_ls = []
        for img_id, fas_surface_dice, epi_surface_dice in surface_dice_scores:
            surface_dice_scores_ls.append([img_id, "fas", fas_surface_dice])
            surface_dice_scores_ls.append([img_id, "epi", epi_surface_dice])
        surface_dice_df = pd.DataFrame(
            surface_dice_scores_ls, columns=["id", "cls", "surface_dice"]
        )
        surface_dice_df.to_csv(
            res_dir / f"surface_dice_scores_{timestamp}.csv", index=False
        )

    frame_surface_dice_scores = results.get("frame_surface_dice")
    if frame_surface_dice_scores:
        frame_surface_dice_df = pd.DataFrame(
            frame_surface_dice_scores,
            columns=["id", "slice_a", "slice_b", "fas", "epi"],
        )
        frame_surface_dice_df = frame_surface_dice_df.melt(
            id_vars=["id", "slice_a", "slice_b"],
            value_vars=["fas", "epi"],
            var_name="cls",
            value_name="surface_dice",
        )
        frame_surface_dice_df.to_csv(
            res_dir / f"frame_surface_dice_scores_{timestamp}.csv", index=False
        )

    asd = results.get("asd")
    if asd:
        asd_df = pd.DataFrame(asd, columns=["id", "fas", "epi"])
        asd_df.to_csv(res_dir / f"avg_surface_distance_{timestamp}.csv", index=False)


@click.command()
@click.option("--model", help="Model name", multiple=True, required=True)
@click.option(
    "--metric",
    "metrics",
    multiple=True,
    type=click.Choice(SUPPORTED_METRICS),
    default=DEFAULT_METRICS,
    show_default=True,
    help="Metrics to compute.",
)
def main(model, metrics):
    model_ls = list(model)
    requested_metrics = list(metrics) or list(DEFAULT_METRICS)

    for m in model_ls:
        sanity_check(m)

    for m in model_ls:
        click.echo(f"Calculating metrics for {m}")
        calculate_metric(m, requested_metrics)


if __name__ == "__main__":
    main()
