from pathlib import Path

import numpy as np
import pandas as pd
from skimage.segmentation import relabel_sequential
from skimage.metrics import hausdorff_distance
from skimage.morphology import label
from tifffile import imread
import datetime
import click
from tqdm import tqdm


def intersection_over_union(ground_truth, prediction):
    # Count objects
    true_objects = len(np.unique(ground_truth))
    pred_objects = len(np.unique(prediction))

    # Compute intersection
    h = np.histogram2d(
        ground_truth.flatten(), prediction.flatten(), bins=(true_objects, pred_objects)
    )
    intersection = h[0]

    # Area of objects
    area_true = np.histogram(ground_truth, bins=true_objects)[0]
    area_pred = np.histogram(prediction, bins=pred_objects)[0]

    # Calculate union
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]

    # Compute Intersection over Union
    union[union == 0] = 1e-9
    IOU = intersection / union

    return IOU


def measures_at(threshold, IOU):
    matches = IOU > threshold

    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Extra objects
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects

    assert np.all(np.less_equal(true_positives, 1))
    assert np.all(np.less_equal(false_positives, 1))
    assert np.all(np.less_equal(false_negatives, 1))

    TP, FP, FN = (
        np.sum(true_positives),
        np.sum(false_positives),
        np.sum(false_negatives),
    )

    f1 = 2 * TP / (2 * TP + FP + FN + 1e-9)
    official_score = TP / (TP + FP + FN + 1e-9)

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)

    return f1, TP, FP, FN, official_score, precision, recall


def compute_instance_metric(ground_truth, prediction, results, image_name, slice_idx):
    # Compute IoU
    IOU = intersection_over_union(ground_truth, prediction)

    tmp_res = []

    # For each ground truth object, find the best matching prediction
    matched_pairs = np.argwhere(IOU > 0.7)
    for pair in matched_pairs:
        pair_iou = IOU[pair[0], pair[1]]
        gt_obj = ground_truth == (pair[0] + 1)
        gt_area = np.sum(gt_obj)
        pred_obj = prediction == (pair[1] + 1)
        pred_area = np.sum(pred_obj)
        hd = hausdorff_distance(gt_obj, pred_obj, method="modified")

        res = {
            "id": image_name,
            "slice": slice_idx,
            "pair_iou": pair_iou,
            "pair_mhd": hd,
            "gt_area": gt_area,
            "pred_area": pred_area,
        }
        tmp_res.append(res)

    results = pd.concat([results, pd.DataFrame(tmp_res)])

    return results


# Compute Average Precision for all IoU thresholds
def compute_af1_results(ground_truth, prediction, results, image_name, slice_idx):
    # Compute IoU
    IOU = intersection_over_union(ground_truth, prediction)
    if IOU.shape[0] > 0:
        jaccard = np.max(IOU, axis=0).mean()
    else:
        jaccard = 0.0

    # Calculate F1 score at all thresholds
    for t in np.arange(0.3, 1.0, 0.05):
        f1, tp, fp, fn, os, prec, rec = measures_at(t, IOU)
        res = {
            "id": image_name,
            "slice": slice_idx,
            "th": t,
            "f1": f1,
            "jaccard": jaccard,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "total": tp + fn,
            "official_score": os,
            "precision": prec,
            "recall": rec,
        }
        row = len(results)
        results.loc[row] = res

    return results


def get_false_negatives(
    ground_truth, prediction, results, image_name, slice_idx, threshold=0.7
):
    # Compute IoU
    IOU = intersection_over_union(ground_truth, prediction)

    true_objects = len(np.unique(ground_truth))
    if true_objects <= 1:
        return results

    area_true = np.histogram(ground_truth, bins=true_objects)[0][1:]
    true_objects -= 1

    # Identify False Negatives
    matches = IOU > threshold
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects

    data = np.asarray(
        [
            area_true.copy(),
            np.array(false_negatives, dtype=np.int32),
            np.repeat(slice_idx, true_objects),
        ]
    )

    df_tmp = pd.DataFrame(data=data.T, columns=["area", "missed", "slice"])
    df_tmp["id"] = image_name
    results = pd.concat([results, df_tmp], ignore_index=True, sort=False)

    return results


# Count the number of splits and merges
def get_splits_and_merges(ground_truth, prediction, results, image_name, slice_idx):
    # Compute IoU
    IOU = intersection_over_union(ground_truth, prediction)

    matches = IOU > 0.1
    merges = np.sum(matches, axis=0) > 1
    splits = np.sum(matches, axis=1) > 1
    r = {
        "id": image_name,
        "slice": slice_idx,
        "merges": np.sum(merges),
        "splits": np.sum(splits),
    }
    results.loc[len(results) + 1] = r
    return results


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
    instance_metrics = pd.DataFrame()
    results = pd.DataFrame(
        columns=[
            "id",
            "slice",
            "th",
            "f1",
            "jaccard",
            "tp",
            "fp",
            "fn",
            "total",
            "official_score",
            "precision",
            "recall",
        ]
    )
    false_negatives = pd.DataFrame(columns=["id", "slice", "area", "missed"])
    splits_merges = pd.DataFrame(columns=["id", "slice", "merges", "splits"])

    for gt_file, pred_file in tqdm(zip(gt_files, pred_files), total=len(gt_files)):
        gt_arr = imread(gt_file) == 1  # fascicle only
        pred_arr = imread(pred_file) == 1  # fascicle only
        assert gt_arr.shape == pred_arr.shape, "Shape mismatch"

        # extract four representative slices
        step_size = gt_arr.shape[0] // 8
        for i in range(0, gt_arr.shape[0], step_size):
            gt_slice = gt_arr[i]
            pred_slice = pred_arr[i]

            gt_slice = label(gt_slice)
            pred_slice = label(pred_slice)

            # Relabel objects
            ground_truth = relabel_sequential(gt_slice)[0]
            prediction = relabel_sequential(pred_slice)[0]

            # Compute metrics
            instance_metrics = compute_instance_metric(
                ground_truth,
                prediction,
                instance_metrics,
                gt_file.stem,
                i,
            )

            results = compute_af1_results(
                ground_truth, prediction, results, gt_file.stem, i
            )

            false_negatives = get_false_negatives(
                ground_truth, prediction, false_negatives, gt_file.stem, i
            )

            splits_merges = get_splits_and_merges(
                ground_truth, prediction, splits_merges, gt_file.stem, i
            )

    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    instance_metrics.to_csv(
        res_dir / f"instance_seg_metrics_{timestamp}.csv", index=False
    )
    results.to_csv(res_dir / f"detection_metrics_{timestamp}.csv", index=False)
    false_negatives.to_csv(res_dir / f"false_negatives_{timestamp}.csv", index=False)
    splits_merges.to_csv(res_dir / f"splits_merges_{timestamp}.csv", index=False)


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
