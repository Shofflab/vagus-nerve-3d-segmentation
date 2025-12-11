import datetime
import warnings
from pathlib import Path

import click
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from tifffile import imread
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)


def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v * s) / np.sum(s)


def clDice(v_p, v_lm, num_class=3, downsample=2):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    # Compute the skeleton of the predicted and ground truth images
    v_p = v_p[:, ::downsample, ::downsample]
    v_lm = v_lm[:, ::downsample, ::downsample]
    res = []

    for i in range(num_class):
        if i == 0:
            continue
        elif i == 1:
            v_p_1 = v_p == i
            v_l_1 = v_lm == i

            t_prec = cl_score(v_p_1, skeletonize(v_l_1))
            t_sens = cl_score(v_l_1, skeletonize(v_p_1))
            res.append(2 * t_prec * t_sens / (t_prec + t_sens))

        elif i == 2:
            v_p_2 = v_p == i
            v_l_2 = v_lm == i

            t_prec = cl_score(v_p_2, skeletonize(v_l_2))
            t_sens = cl_score(v_l_2, skeletonize(v_p_2))
            res.append(2 * t_prec * t_sens / (t_prec + t_sens))

    return res


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

    # metrics storage
    cl_dice_scores = []

    for gt_file, pred_file in tqdm(zip(gt_files, pred_files), total=len(gt_files)):
        gt_arr = imread(gt_file)
        pred_arr = imread(pred_file)

        assert gt_arr.shape == pred_arr.shape, "Shape mismatch"

        cl_dice = clDice(pred_arr, gt_arr, num_class=3, downsample=1)
        cl_dice_scores.append([gt_file.stem, cl_dice])

    timestamp = datetime.datetime.now().strftime("%Y%m%d")

    # Save results
    if cl_dice_scores:
        cl_dice_scores_ls = []
        for i in range(len(cl_dice_scores)):
            img_id = cl_dice_scores[i]
            fas_cldice = img_id[1][0]
            epi_cldice = img_id[1][1]
            cl_dice_scores_ls.append([img_id[0], "fas", fas_cldice])
            cl_dice_scores_ls.append([img_id[0], "epi", epi_cldice])

        df = pd.DataFrame(cl_dice_scores_ls, columns=["img_id", "cls", "cl_dice"])
        df.to_csv(res_dir / f"cl_dice_scores_{timestamp}.csv", index=False)


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
