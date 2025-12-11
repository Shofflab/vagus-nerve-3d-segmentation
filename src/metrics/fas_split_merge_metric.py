from pathlib import Path
from typing import Tuple, TypedDict

import click
import cv2
import igraph as ig
import numpy as np
import pandas as pd
import zarr
from tifffile import imread
from tqdm import tqdm

OVERLAP_THRESHOLD = 0.05
SPLIT_MERGE_THRESHOLD = 0.85
AREA_WEIGHT = 0.5
CENTROID_WEIGHT = 0.5
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"


class NodeProperty(TypedDict):
    frame: int
    label: int
    track_id: int
    tree_id: int
    area: float
    equivalent_diameter_area: float
    centroid_0: float
    centroid_1: float
    major_axis: float
    minor_axis: float
    angle: float


class EdgeProperty(TypedDict):
    frame: Tuple[int, int]
    identity: bool
    split: bool
    merge: bool
    direction_0: float
    direction_1: float


def conf_mat(y_true, y_pred, N=None):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    if N is None:
        N = max(max(y_true), max(y_pred)) + 1
    y = N * y_true + y_pred
    y = np.bincount(y, minlength=N * N)
    y = y.reshape(N, N)
    return y


def overlap_metric(conf_mat, label1, label2):
    """
    Calculates the overlap metric between two labels based on the given confusion matrix.

    Parameters:
    conf_mat (numpy.ndarray): The confusion matrix.
    label1 (int): The first label.
    label2 (int): The second label.

    Returns:
    float: The overlap metric between the two labels.
    """
    overlap = conf_mat[label1, label2]
    b1_sum = np.sum(conf_mat[label1, :])
    b2_sum = np.sum(conf_mat[:, label2])

    return (overlap / 2) * (1 / b1_sum + 1 / b2_sum)


def find_exact_match(array):
    """
    Find exact matches in the given array.

    Parameters:
    array (ndarray): The input array.

    Returns:
    ndarray or None: The array containing the exact matches, or None if no exact matches are found.
    """
    _, idx_1, count_1 = np.unique(array[:, 0], return_index=True, return_counts=True)
    _, idx_2, count_2 = np.unique(array[:, 1], return_index=True, return_counts=True)
    match = np.intersect1d(idx_1[count_1 == 1], idx_2[count_2 == 1])

    return array[match] if len(match) > 0 else None


def find_split_merge(array):
    """
    Finds split and merge arrays based on the input array.

    Parameters:
    array (numpy.ndarray): The input array.

    Returns:
    tuple: A tuple containing the split and merge arrays. Each array is a list of tuples, where each tuple contains the element and the corresponding rows.

    Example:
    >>> array = np.array([[1, 2], [1, 3], [2, 4], [2, 5]])
    >>> find_split_merge(array)
    ([(1, [2, 3])], [(2, [4, 5])])
    """
    # a tuple storing the split and merge arrays
    res = ([], [])

    for col in range(len(res)):
        values, counts = np.unique(array[:, col], return_counts=True)
        rep_elements = values[counts > 1]

        for element in rep_elements:
            rows = array[array[:, col] == element]
            res[col].append((element, rows[:, 1 - col]))

    split = None if len(res[0]) == 0 else res[0]
    merge = None if len(res[1]) == 0 else res[1]

    return split, merge


def area_metric(conf_mat, parent_idx, child_idx):
    """
    Calculates the area metric between a parent region and a child region based on the confusion matrix.

    Parameters:
    conf_mat (numpy.ndarray): The confusion matrix representing the region segmentation.
    parent_idx (int): The index of the parent region.
    child_idx (int): The index of the child region.

    Returns:
    float: The area metric value between the parent and child regions.
    """
    parent_area = np.sum(conf_mat[parent_idx, :])
    child_area = np.sum(conf_mat[:, child_idx])

    A = 1 / (1 + (abs(parent_area - child_area) / min(parent_area, child_area)))
    return A


def centroid_metric(parent_props, parent_idx, child_props, child_idx, img_h, img_w):
    """
    Calculate the centroid metric between a parent region and a child region.

    Parameters:
    - parent_props (dict): Dictionary containing properties of the parent region.
    - parent_idx (int): Index of the parent region.
    - child_props (dict): Dictionary containing properties of the child region.
    - child_idx (int): Index of the child region.
    - img_h (int): Height of the image.
    - img_w (int): Width of the image.

    Returns:
    - C (float): Centroid metric value.
    """
    parent_centroid = (
        parent_props["centroid-0"][parent_idx - 1],
        parent_props["centroid-1"][parent_idx - 1],
    )
    # calculate the average area weighted centroid of all children
    child_centroid = (
        np.average(
            child_props["centroid-0"][child_idx - 1],
            weights=child_props["area"][child_idx - 1],
        ),
        np.average(
            child_props["centroid-1"][child_idx - 1],
            weights=child_props["area"][child_idx - 1],
        ),
    )

    norm = np.sqrt(img_h**2 + img_w**2)
    dist_diff = np.sqrt(
        (parent_centroid[0] - child_centroid[0]) ** 2
        + (parent_centroid[1] - child_centroid[1]) ** 2
    )
    C = 1 / (1 + (dist_diff / norm))

    return C


def select_prop(idx, prop_dict):
    tmp_prop = prop_dict.copy()
    return {k: v[idx] for k, v in tmp_prop.items()}


def compute_confusion_matrix(
    true_labels: np.ndarray, pred_labels: np.ndarray, num_classes: int = None
) -> np.ndarray:
    """
    Computes confusion matrix between two label arrays.
    """
    # Flatten arrays only once
    true_flat = true_labels.ravel()
    pred_flat = pred_labels.ravel()

    # Use max from flattened arrays to avoid redundant operations
    if num_classes is None:
        num_classes = max(true_flat.max(), pred_flat.max()) + 1

    # Use numpy's built-in histogram2d which is faster than bincount
    conf_matrix = np.histogram2d(
        true_flat,
        pred_flat,
        bins=(num_classes, num_classes),
        range=[[0, num_classes], [0, num_classes]],
    )[0]

    return conf_matrix.astype(int)


def calculate_overlap_score(overlap_mat, label1, label2):
    """Calculate overlap score between two labels using confusion matrix."""
    overlap = overlap_mat[label1, label2]
    label1_sum = np.sum(overlap_mat[label1, :])
    label2_sum = np.sum(overlap_mat[:, label2])

    return overlap / min(label1_sum, label2_sum)


def track_fascicles(
    graph: ig.Graph,
    image_sequence: np.ndarray,
    start_frame: int,
    end_frame: int = None,
    image_preprocessor: callable = lambda x: x,
    overlap_threshold: float = OVERLAP_THRESHOLD,
    split_merge_threshold: float = SPLIT_MERGE_THRESHOLD,
    area_weight: float = AREA_WEIGHT,
    centroid_weight: float = CENTROID_WEIGHT,
) -> None:
    """
    Tracks fascicles across frames and builds a graph representation.

    Args:
        graph: Empty igraph Graph to store tracking results
        image_sequence: 3D array of binary segmentation masks
        start_frame: First frame to process
        end_frame: Last frame to process (None for all frames)
        image_preprocessor: Optional function to preprocess each frame
        overlap_threshold: Minimum overlap required between regions
        split_merge_threshold: Threshold for considering split/merge events
        area_weight: Weight for area metric in split/merge calculation
        centroid_weight: Weight for centroid metric in split/merge calculation
    """
    end_frame = len(image_sequence)

    # Pre-allocate tracking values
    track_count = 1
    last_img_tuple = None
    last_frame_vertices = None

    for frame_idx in range(start_frame, end_frame - 1):
        current_frame = image_preprocessor(image_sequence[frame_idx])

        # Skip empty frames efficiently using sum
        if not np.any(current_frame == 1):
            continue

        if last_img_tuple is None:
            tuple_1 = extract_connected_components(current_frame)
        else:
            tuple_1 = last_img_tuple

        next_frame = image_preprocessor(image_sequence[frame_idx + 1])
        if not np.any(next_frame == 1):
            continue

        tuple_2 = extract_connected_components(next_frame)

        # Create vertices for current frame
        label_1, props_1, unique_1 = tuple_1
        label_2, props_2, unique_2 = tuple_2

        vert_ls_2 = [graph.add_vertex() for _ in range(len(unique_2))]

        if last_frame_vertices is None:
            vert_ls_1 = [graph.add_vertex() for _ in range(len(unique_1))]
        else:
            vert_ls_1 = last_frame_vertices

        # Add properties to vertices
        if last_frame_vertices is None:
            for idx, vert in enumerate(vert_ls_1):
                tmp_prop = select_prop(idx, props_1)
                for k, v in tmp_prop.items():
                    vert[k] = v
                vert["frame"] = frame_idx
                vert["track_id"] = int(track_count)
                track_count += 1

        for idx, vert in enumerate(vert_ls_2):
            tmp_prop = select_prop(idx, props_2)
            for k, v in tmp_prop.items():
                vert[k] = v
            vert["frame"] = frame_idx + 1

        # Link frames
        img_h, img_w = label_1.shape
        combs = np.array(np.meshgrid(unique_1, unique_2)).T.reshape(-1, 2)
        overlap_mat = compute_confusion_matrix(
            label_1.ravel(),
            label_2.ravel(),
            num_classes=max(max(unique_1), max(unique_2)) + 1,
        )

        # Find overlapping regions
        overlapping_labels = []
        for comb in combs:
            overlap_score = calculate_overlap_score(overlap_mat, comb[0], comb[1])
            if overlap_score > overlap_threshold:
                overlapping_labels.append(np.array([comb[0], comb[1]]))

        if not overlapping_labels:
            last_frame_vertices = vert_ls_2
            last_img_tuple = tuple_2
            continue

        overlapping_labels = np.stack(overlapping_labels)

        track_edges = find_exact_match(overlapping_labels)
        if track_edges is not None:
            for i, j in track_edges:
                edge = graph.add_edge(vert_ls_1[i - 1], vert_ls_2[j - 1])
                edge["frame"] = [frame_idx, frame_idx + 1]
                edge["identity"] = True

                source = graph.vs[edge.source]
                target = graph.vs[edge.target]

                edge["direction-0"] = target["centroid-0"] - source["centroid-0"]
                edge["direction-1"] = target["centroid-1"] - source["centroid-1"]
                target["track_id"] = source["track_id"]

        split, merge = find_split_merge(overlapping_labels)

        if split is not None:
            for s in split:
                area_score = area_metric(overlap_mat, s[0], s[1])
                centroid_score = centroid_metric(
                    props_1, s[0], props_2, s[1], img_h, img_w
                )
                split_score = (
                    area_score * area_weight + centroid_score * centroid_weight
                )

                if split_score > split_merge_threshold:
                    for target_label in s[1]:
                        edge = graph.add_edge(
                            vert_ls_1[s[0] - 1], vert_ls_2[target_label - 1]
                        )
                        edge["frame"] = [frame_idx, frame_idx + 1]
                        edge["split"] = True

                        source = graph.vs[edge.source]
                        target = graph.vs[edge.target]

                        edge["direction-0"] = (
                            target["centroid-0"] - source["centroid-0"]
                        )
                        edge["direction-1"] = (
                            target["centroid-1"] - source["centroid-1"]
                        )
                        target["track_id"] = int(track_count)
                        track_count += 1

        if merge is not None:
            for m in merge:
                area_score = area_metric(overlap_mat.T, m[0], m[1])
                centroid_score = centroid_metric(
                    props_2, m[0], props_1, m[1], img_h, img_w
                )
                merge_score = (
                    area_score * area_weight + centroid_score * centroid_weight
                )

                if merge_score > split_merge_threshold:
                    for source_label in m[1]:
                        edge = graph.add_edge(
                            vert_ls_1[source_label - 1], vert_ls_2[m[0] - 1]
                        )
                        edge["frame"] = [frame_idx, frame_idx + 1]
                        edge["merge"] = True

                        source = graph.vs[edge.source]
                        target = graph.vs[edge.target]

                        edge["direction-0"] = (
                            target["centroid-0"] - source["centroid-0"]
                        )
                        edge["direction-1"] = (
                            target["centroid-1"] - source["centroid-1"]
                        )
                        target["track_id"] = int(track_count)
                        track_count += 1

        for vert in vert_ls_2:
            if vert["track_id"] is None:
                vert["track_id"] = int(track_count)
                track_count += 1

        last_img_tuple = tuple_2
        last_frame_vertices = vert_ls_2


def extract_connected_components(
    binary_image: np.ndarray, target_value: int = 1
) -> Tuple[np.ndarray, dict, np.ndarray]:
    img = (binary_image == target_value).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img, 1, cv2.CV_32S, connectivity=4
    )

    area = stats[1:, cv2.CC_STAT_AREA]
    equiv_diam = np.sqrt(area / np.pi) * 2
    unique_labels = np.arange(1, num_labels)

    n_components = num_labels - 1
    ellipse_props = {
        "center_x": np.zeros(n_components),
        "center_y": np.zeros(n_components),
        "major_axis": np.zeros(n_components),
        "minor_axis": np.zeros(n_components),
        "angle": np.zeros(n_components),
    }

    contours = []
    for i in range(1, num_labels):
        mask = (labels == i).astype(np.uint8)
        contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.append(contour[0] if contour else None)

    for i, contour in enumerate(contours):
        if contour is not None and len(contour) > 5:
            (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
            ellipse_props["center_x"][i] = x
            ellipse_props["center_y"][i] = y
            ellipse_props["major_axis"][i] = MA
            ellipse_props["minor_axis"][i] = ma
            ellipse_props["angle"][i] = angle

    props = {
        "label": unique_labels,
        "area": area,
        "equivalent_diameter": equiv_diam,
        "centroid-0": centroids[1:, 0],
        "centroid-1": centroids[1:, 1],
        "ellipse_center-0": ellipse_props["center_x"],
        "ellipse_center-1": ellipse_props["center_y"],
        "ellipse_major_axis": ellipse_props["major_axis"],
        "ellipse_minor_axis": ellipse_props["minor_axis"],
        "ellipse_angle": ellipse_props["angle"],
    }

    return labels, props, unique_labels


def process_fas_connectivity(g):
    fas_connectivity_df_ls = []
    edge_df = pd.DataFrame({attr: g.es[attr] for attr in g.edge_attributes()})
    edge_df["frame"] = [int(g.vs[e.source]["frame"]) for e in g.es]
    edge_df["source"] = [e.source for e in g.es]
    edge_df["target"] = [e.target for e in g.es]
    fas_connectivity_df_ls.append(edge_df)

    fas_connectivity_df = pd.concat(fas_connectivity_df_ls)

    return fas_connectivity_df


def process_segmentation_file(seg_file: Path) -> dict:
    name = seg_file.stem.split(".")[0]
    tracking_graph = ig.Graph(directed=True)
    segmentation_store = imread(seg_file, aszarr=True)
    segmentation_data = zarr.open(segmentation_store, mode="r")
    track_fascicles(
        tracking_graph,
        segmentation_data,
        0,
        None,
        image_preprocessor=lambda x: (x != 0).astype(np.uint8),
    )
    fas_connectivity_df = process_fas_connectivity(tracking_graph)
    if "split" in fas_connectivity_df.columns:
        split_df = fas_connectivity_df[fas_connectivity_df["split"] == True]
        split_df = split_df.drop_duplicates(subset=["frame", "source"])
        num_splits = len(split_df)
    else:
        num_splits = 0

    if "merge" in fas_connectivity_df.columns:
        merge_df = fas_connectivity_df[fas_connectivity_df["merge"] == True]
        merge_df = merge_df.drop_duplicates(subset=["frame", "target"])
        num_merges = len(merge_df)
    else:
        num_merges = 0

    return {
        "id": name,
        "split": num_splits,
        "merge": num_merges,
    }


def discover_segmentation_files(model):
    seg_dir = DATA_DIR / "gt" if model == "gt" else DATA_DIR / model / "pred"
    if not seg_dir.exists():
        raise FileNotFoundError(f"Segmentation directory not found: {seg_dir}")
    seg_files = sorted(seg_dir.rglob("*.tif*"))
    if not seg_files:
        raise FileNotFoundError(f"No segmentation files found in {seg_dir}")
    return seg_files


@click.command()
@click.option("--model", help="Model name", multiple=True, required=True)
def main(model):
    model_ls = list(model)
    for m in model_ls:
        segmentation_files = discover_segmentation_files(m)
        df_ls = []
        for seg_file in tqdm(segmentation_files, desc=f"Processing {m}", unit="file"):
            df_ls.append(process_segmentation_file(seg_file))
        res_dir = RESULTS_DIR / m
        res_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(df_ls)
        df.to_csv(res_dir / "split_merge_events.csv", index=False)


if __name__ == "__main__":
    main()
