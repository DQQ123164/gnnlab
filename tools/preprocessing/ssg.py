"""Preprocess OpenPose BODY_25 poses for static skeleton classification."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .common import (
    LOGGER,
    SplitName,
    atomic_save_npy,
    count_labels,
    count_warning_types,
    relative_to,
    require_dir,
    write_json,
)


SSG_CLASS_MAP: dict[str, int] = {
    "sitting": 0,
    "standing": 1,
}

SSG_SCALE_EDGES: tuple[tuple[int, int], ...] = (
    (1, 8),
    (1, 2),
    (1, 5),
    (2, 5),
    (8, 9),
    (8, 12),
    (9, 12),
)

BODY25_JOINT_NAMES: list[str] = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel",
]

BODY25_EDGES: list[tuple[int, int]] = [
    (1, 8),
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (8, 9),
    (9, 10),
    (10, 11),
    (8, 12),
    (12, 13),
    (13, 14),
    (1, 0),
    (0, 15),
    (15, 17),
    (0, 16),
    (16, 18),
    (14, 19),
    (19, 20),
    (14, 21),
    (11, 22),
    (22, 23),
    (11, 24),
]


@dataclass(frozen=True)
class SsgSampleRecord:
    sample_name: str
    split: SplitName
    class_name: str
    label: int
    source_path: str
    people_count: int
    selected_person_index: int | None
    selected_confidence_sum: float
    valid_joint_count: int
    center: list[float]
    scale: float
    warnings: list[str]


@dataclass(frozen=True)
class SsgSplitSummary:
    split: SplitName
    data_file: str
    label_file: str
    source_count: int
    sample_count: int
    skipped_count: int
    data_shape: tuple[int, ...]
    dtype: str
    label_counts: dict[str, int]
    warnings: dict[str, int]


def preprocess_ssg(
    source_dir: Path,
    output_dir: Path,
    overwrite: bool,
    min_confidence: float,
) -> dict[str, Any]:
    require_dir(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[SsgSplitSummary] = []
    for split in ("train", "test"):
        split_name: SplitName = split  # type: ignore[assignment]
        samples, skipped_records = collect_ssg_samples(
            source_dir,
            split_name,
            min_confidence,
        )
        if not samples:
            raise FileNotFoundError(
                f"No usable ssg {split} JSON files found under {source_dir}"
            )

        norm_data = np.stack([sample[1] for sample in samples]).astype(np.float32)
        records = [sample[2] for sample in samples]
        labels = np.asarray([record.label for record in records], dtype=np.int64)

        data_dst = output_dir / f"{split}_data.npy"
        label_dst = output_dir / f"{split}_label.npy"
        atomic_save_npy(data_dst, norm_data, overwrite=overwrite)
        atomic_save_npy(label_dst, labels, overwrite=overwrite)

        summaries.append(
            SsgSplitSummary(
                split=split_name,
                data_file=relative_to(data_dst, output_dir),
                label_file=relative_to(label_dst, output_dir),
                source_count=len(records) + len(skipped_records),
                sample_count=len(records),
                skipped_count=len(skipped_records),
                data_shape=tuple(norm_data.shape),
                dtype=str(norm_data.dtype),
                label_counts=count_labels(labels.tolist()),
                warnings=count_warning_types(records + skipped_records),
            )
        )

    metadata = {
        "task": "static_skeleton_classification",
        "format": {
            "data": "float32 numpy array shaped (N, V, C)",
            "channels": ["x_normalized", "y_normalized", "confidence"],
            "label_npy": "int64 labels aligned with data rows",
            "normalization": {
                "center": "mean of valid MidHip/RHip/LHip; fallback to upper torso or all valid joints",
                "rotation": "align hip-center-to-Neck axis vertically; fallback to shoulder axis or image axes",
                "scale": "median valid torso-edge length; fallback to bbox max side or 1.0",
                "invalid_joint": "x/y/confidence are set to 0",
                "invalid_sample": "samples without any valid BODY_25 joint are skipped",
                "min_confidence": min_confidence,
            },
            "person_selection": "highest sum of BODY_25 joint confidence",
        },
        "class_map": SSG_CLASS_MAP,
        "body25": {
            "joint_names": BODY25_JOINT_NAMES,
            "edges": BODY25_EDGES,
        },
        "splits": [asdict(summary) for summary in summaries],
    }
    write_json(output_dir / "metadata.json", metadata, overwrite=overwrite)
    if overwrite:
        remove_obsolete_ssg_outputs(output_dir)
    LOGGER.info(
        "ssg: converted %d compact splits (%d usable samples, %d skipped)",
        len(summaries),
        sum(summary.sample_count for summary in summaries),
        sum(summary.skipped_count for summary in summaries),
    )
    return metadata


def remove_obsolete_ssg_outputs(output_dir: Path) -> None:
    obsolete_names = (
        "train_data_raw.npy",
        "test_data_raw.npy",
        "train_label.pkl",
        "test_label.pkl",
        "train_samples.txt",
        "test_samples.txt",
        "train_manifest.jsonl",
        "test_manifest.jsonl",
        "body25_adjacency.npy",
        "body25_edge_index.npy",
        "body25_graph.json",
    )
    for name in obsolete_names:
        path = output_dir / name
        if path.exists() or path.is_symlink():
            path.unlink()


def collect_ssg_samples(
    source_dir: Path,
    split: SplitName,
    min_confidence: float,
) -> tuple[
    list[tuple[np.ndarray, np.ndarray, SsgSampleRecord]],
    list[SsgSampleRecord],
]:
    samples: list[tuple[np.ndarray, np.ndarray, SsgSampleRecord]] = []
    skipped_records: list[SsgSampleRecord] = []
    for class_name, label in sorted(SSG_CLASS_MAP.items(), key=lambda item: item[1]):
        split_dir = source_dir / class_name / split
        require_dir(split_dir)
        for json_path in sorted(split_dir.glob("*.json")):
            raw, normalized, record = parse_ssg_json(
                json_path=json_path,
                source_dir=source_dir,
                split=split,
                class_name=class_name,
                label=label,
                min_confidence=min_confidence,
            )
            if record.valid_joint_count == 0:
                skipped_records.append(record)
                continue
            samples.append((raw, normalized, record))
    return samples, skipped_records


def parse_ssg_json(
    json_path: Path,
    source_dir: Path,
    split: SplitName,
    class_name: str,
    label: int,
    min_confidence: float,
) -> tuple[np.ndarray, np.ndarray, SsgSampleRecord]:
    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    people = payload.get("people") or []
    warnings: list[str] = []
    selected_index: int | None = None
    selected_confidence_sum = 0.0
    raw = np.zeros((25, 3), dtype=np.float32)

    if not people:
        warnings.append("no_person")
    else:
        selected_index, selected = select_person(people)
        keypoints = selected.get("pose_keypoints_2d") or []
        selected_confidence_sum = confidence_sum(keypoints)
        if len(keypoints) != 75:
            warnings.append("invalid_keypoint_length")
        else:
            raw = np.asarray(keypoints, dtype=np.float32).reshape(25, 3)
        if len(people) > 1:
            warnings.append("multi_person")

    normalized, center, scale, valid_count, norm_warnings = normalize_body25(
        raw,
        min_confidence=min_confidence,
    )
    warnings.extend(norm_warnings)

    record = SsgSampleRecord(
        sample_name=json_path.stem,
        split=split,
        class_name=class_name,
        label=label,
        source_path=relative_to(json_path, source_dir),
        people_count=len(people),
        selected_person_index=selected_index,
        selected_confidence_sum=float(selected_confidence_sum),
        valid_joint_count=valid_count,
        center=[float(center[0]), float(center[1])],
        scale=float(scale),
        warnings=warnings,
    )
    return raw, normalized, record


def select_person(people: Sequence[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    scores = [
        confidence_sum(person.get("pose_keypoints_2d") or [])
        for person in people
    ]
    selected_index = int(np.argmax(np.asarray(scores, dtype=np.float32)))
    return selected_index, people[selected_index]


def confidence_sum(keypoints: Sequence[float]) -> float:
    if len(keypoints) < 3:
        return 0.0
    return float(sum(float(value) for value in keypoints[2::3]))


def normalize_body25(
    raw: np.ndarray,
    min_confidence: float,
) -> tuple[np.ndarray, np.ndarray, float, int, list[str]]:
    warnings: list[str] = []
    normalized = np.zeros_like(raw, dtype=np.float32)

    valid = valid_joint_mask(raw, min_confidence)
    valid_count = int(valid.sum())
    if valid_count == 0:
        warnings.append("no_valid_joints")
        return normalized, np.asarray([0.0, 0.0], dtype=np.float32), 1.0, 0, warnings

    center, center_source = estimate_center(raw, valid)
    aligned_xy, rotation_source = align_body25(raw, valid, center, center_source)
    scale, scale_source = estimate_scale(raw, valid)
    if center_source != "hips":
        warnings.append(f"center_from_{center_source}")
    if rotation_source != "torso":
        warnings.append(f"rotation_from_{rotation_source}")
    if scale_source != "torso_edges":
        warnings.append(f"scale_from_{scale_source}")

    normalized[valid, :2] = aligned_xy[valid] / scale
    normalized[valid, 2] = raw[valid, 2]
    return normalized, center.astype(np.float32), float(scale), valid_count, warnings


def valid_joint_mask(raw: np.ndarray, min_confidence: float) -> np.ndarray:
    nonzero_xy = np.logical_or(raw[:, 0] != 0.0, raw[:, 1] != 0.0)
    confident = raw[:, 2] > min_confidence
    return np.logical_and(nonzero_xy, confident)


def estimate_center(
    raw: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, str]:
    for source, joint_indices in (
        ("hips", (8, 9, 12)),
        ("upper_torso", (1, 2, 5)),
    ):
        selected = [index for index in joint_indices if valid[index]]
        if selected:
            return raw[selected, :2].mean(axis=0), source
    return raw[valid, :2].mean(axis=0), "visible_mean"


def align_body25(
    raw: np.ndarray,
    valid: np.ndarray,
    center: np.ndarray,
    center_source: str,
) -> tuple[np.ndarray, str]:
    shifted = raw[:, :2] - center

    if center_source == "hips" and valid[1]:
        upward = raw[1, :2] - center
        axis_length = float(np.linalg.norm(upward))
        if axis_length > 1e-6:
            upward /= axis_length
            horizontal = np.asarray([-upward[1], upward[0]], dtype=np.float32)
            downward = -upward
            return np.column_stack((shifted @ horizontal, shifted @ downward)), "torso"

    if valid[2] and valid[5]:
        horizontal = raw[5, :2] - raw[2, :2]
        axis_length = float(np.linalg.norm(horizontal))
        if axis_length > 1e-6:
            horizontal /= axis_length
            downward = np.asarray([-horizontal[1], horizontal[0]], dtype=np.float32)
            return np.column_stack((shifted @ horizontal, shifted @ downward)), "shoulders"

    return shifted, "image_axes"


def estimate_scale(raw: np.ndarray, valid: np.ndarray) -> tuple[float, str]:
    edge_lengths = [
        float(np.linalg.norm(raw[first, :2] - raw[second, :2]))
        for first, second in SSG_SCALE_EDGES
        if valid[first]
        and valid[second]
        and np.linalg.norm(raw[first, :2] - raw[second, :2]) > 1e-6
    ]
    if edge_lengths:
        return float(np.median(np.asarray(edge_lengths))), "torso_edges"

    points = raw[valid, :2]
    width_height = points.max(axis=0) - points.min(axis=0)
    bbox_scale = float(np.max(width_height))
    if bbox_scale > 1e-6:
        return bbox_scale, "bbox"

    return 1.0, "unit"
