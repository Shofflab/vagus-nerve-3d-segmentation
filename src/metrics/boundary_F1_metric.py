from itertools import pairwise
from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
from tifffile import imread
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"


def get_contours(image, target_class):
    image = np.where(image == target_class, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_out = []
    for contour in contours:
        contours_out.extend(contour.reshape(-1, 2))
    return np.array(contours_out, dtype=np.float32)


def calc_precision_recall(contours_a, contours_b, threshold):
    if contours_b.size == 0 or contours_a.size == 0:
        return 0, 0, max(len(contours_b), 1)

    # Compute squared distances using broadcasting
    d = np.sum((contours_a[:, None, :] - contours_b[None, :, :]) ** 2, axis=2)
    hits = np.any(d < threshold**2, axis=0)
    top_count = np.sum(hits)

    precision_recall = top_count / len(contours_b)
    return precision_recall, top_count, len(contours_b)


def bfscore(gt_, pr_, threshold=2):
    classes = np.unique(np.concatenate((gt_, pr_)))
    bfscores = np.full((classes.max() + 1,), np.nan)

    for target_class in classes:
        if target_class == 0:
            continue

        contours_gt = get_contours(gt_, target_class)
        contours_pr = get_contours(pr_, target_class)

        precision, _, _ = calc_precision_recall(contours_gt, contours_pr, threshold)
        recall, _, _ = calc_precision_recall(contours_pr, contours_gt, threshold)

        # Calculate F1 score
        if recall + precision > 0:
            f1 = 2 * recall * precision / (recall + precision)
        else:
            f1 = np.nan
        bfscores[target_class] = f1

    return bfscores[1:]


def load_prediction_files(model):
    if model == "gt":
        pred_dir = DATA_DIR / "gt"
    else:
        pred_dir = DATA_DIR / model / "pred"
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    files = sorted(pred_dir.glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No prediction files found in {pred_dir}")
    return files


def write_results(model, rows):
    model_results_dir = RESULTS_DIR / model
    model_results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["id", "slice_a", "slice_b", "fas", "epi"])
    df.to_csv(model_results_dir / "bfscores.csv", index=False)


@click.command()
@click.option("--model", help="Model name", multiple=True, required=True)
def main(model):
    model_ls = list(model)
    for m in model_ls:
        bfscore_res = []
        for img_path in tqdm(load_prediction_files(m)):
            img = imread(img_path)
            for i, j in pairwise(range(len(img))):
                bfscores = bfscore(img[i], img[j], threshold=1)
                bfscore_res.append([img_path.stem, i, j, *bfscores])
        write_results(m, bfscore_res)


if __name__ == "__main__":
    main()
