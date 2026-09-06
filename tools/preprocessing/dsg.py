"""Preprocess NTU RGB+D sequences for dynamic skeleton classification."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .common import (
    LOGGER,
    ProgressBar,
    SplitName,
    atomic_save_npy,
    count_labels,
    count_warning_types,
    prepare_destination,
    relative_to,
    require_dir,
    write_json,
    write_jsonl,
    write_pickle,
    write_text,
)


DSG_CLASS_MAP: dict[str, int] = {
    "A059_walking_towards_each_other": 0,
    "A030_typing_on_a_keyboard": 1,
    "A016_wear_a_shoe": 2,
    "A005_drop": 3,
    "A027_jump_up": 4,
}

DSG_ACTION_LABELS: dict[str, int] = {
    "A059": 0,
    "A030": 1,
    "A016": 2,
    "A005": 3,
    "A027": 4,
}

DSG_MAX_FRAMES = 300
DSG_NUM_JOINTS = 25
DSG_MAX_BODIES = 2
DSG_FILENAME_PATTERN = re.compile(
    r"S\d{3}C(?P<camera>\d{3})P(?P<subject>\d{3})"
    r"R\d{3}(?P<action>A\d{3})\.skeleton$",
    re.IGNORECASE,
)
DSG_TRAINING_SUBJECTS = {
    1,
    2,
    4,
    5,
    8,
    9,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    25,
    27,
    28,
    31,
    34,
    35,
    38,
}
DSG_TRAINING_CAMERAS = {2, 3}
DSG_EXPECTED_SPLIT_COUNTS = {
    "xsub": {"train": 3341, "test": 1372},
    "xview": {"train": 3134, "test": 1579},
}


@dataclass(frozen=True)
class DsgSplitSummary:
    protocol: str
    split: SplitName
    data_file: str
    label_file: str
    data_shape: tuple[int, ...]
    dtype: str
    sample_count: int
    label_counts: dict[str, int]
    source_bytes: int
    manifest_file: str
    warnings: dict[str, int]


@dataclass(frozen=True)
class DsgSampleRecord:
    sample_name: str
    protocol: str
    split: SplitName
    action_id: str
    label: int
    source_path: str
    frame_count: int
    stored_frame_count: int
    max_body_count: int
    empty_frame_count: int
    dropped_body_instances: int
    warnings: list[str]


@dataclass(frozen=True)
class DsgSourceSample:
    path: Path
    action_id: str
    label: int
    subject: int
    camera: int


def preprocess_dsg(
    source_dir: Path,
    output_dir: Path,
    overwrite: bool,
    show_progress: bool,
) -> dict[str, Any]:
    require_dir(source_dir)
    protocol_splits = discover_materialized_protocols(source_dir)
    if protocol_splits is None:
        raw_dir = find_dsg_raw_dir(source_dir)
        samples = discover_dsg_samples(raw_dir)
        protocol_splits = build_official_protocols(samples)
        source_layout = "single raw directory"
    else:
        raw_dir = source_dir
        source_layout = "materialized xsub/xview train/test directories"
    validate_official_protocols(protocol_splits)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[DsgSplitSummary] = []
    for protocol, splits in protocol_splits.items():
        protocol_out = output_dir / protocol
        protocol_out.mkdir(parents=True, exist_ok=True)

        for split in ("train", "test"):
            split_name: SplitName = split  # type: ignore[assignment]
            split_samples = splits[split_name]
            skeleton_paths = [sample.path for sample in split_samples]

            data_dst = protocol_out / f"{split}_data.npy"
            label_dst = protocol_out / f"{split}_label.pkl"
            label_npy_dst = protocol_out / f"{split}_label.npy"
            samples_dst = protocol_out / f"{split}_samples.txt"
            manifest_dst = protocol_out / f"{split}_manifest.jsonl"
            destinations = (
                data_dst,
                label_dst,
                label_npy_dst,
                samples_dst,
                manifest_dst,
            )
            for destination in destinations:
                prepare_destination(destination, overwrite)

            labels = [sample.label for sample in split_samples]
            action_ids = [sample.action_id for sample in split_samples]
            sample_names = [path.name for path in skeleton_paths]
            records = write_dsg_array(
                skeleton_paths=skeleton_paths,
                action_ids=action_ids,
                labels=labels,
                protocol=protocol,
                split=split_name,
                source_dir=source_dir,
                destination=data_dst,
                overwrite=overwrite,
                show_progress=show_progress,
            )
            atomic_save_npy(
                label_npy_dst,
                np.asarray(labels, dtype=np.int64),
                overwrite=overwrite,
            )
            write_text(
                samples_dst,
                "\n".join(sample_names) + "\n",
                overwrite=overwrite,
            )
            write_pickle(
                label_dst,
                (sample_names, labels),
                overwrite=overwrite,
            )
            write_jsonl(
                manifest_dst,
                [asdict(record) for record in records],
                overwrite=overwrite,
            )

            shape = (
                len(skeleton_paths),
                3,
                DSG_MAX_FRAMES,
                DSG_NUM_JOINTS,
                DSG_MAX_BODIES,
            )
            summaries.append(
                DsgSplitSummary(
                    protocol=protocol,
                    split=split_name,
                    data_file=relative_to(data_dst, output_dir),
                    label_file=relative_to(label_dst, output_dir),
                    data_shape=shape,
                    dtype="float32",
                    sample_count=len(labels),
                    label_counts=count_labels(labels),
                    source_bytes=sum(path.stat().st_size for path in skeleton_paths),
                    manifest_file=relative_to(manifest_dst, output_dir),
                    warnings=count_warning_types(records),
                )
            )

    metadata = {
        "task": "dynamic_skeleton_classification",
        "source_format": "NTU RGB+D .skeleton text files",
        "source_layout": source_layout,
        "protocols": {
            "xsub": "official NTU RGB+D 60 cross-subject split",
            "xview": "official NTU RGB+D 60 cross-view split",
        },
        "format": {
            "data": "float32 numpy array shaped (N, C, T, V, M)",
            "channels": ["x", "y", "z"],
            "max_frames": DSG_MAX_FRAMES,
            "num_joints": DSG_NUM_JOINTS,
            "max_bodies": DSG_MAX_BODIES,
            "padding": "missing frames and bodies are zero-filled",
            "overflow": "frames after max_frames and bodies after max_bodies are discarded",
            "label_pkl": "pickle tuple (sample_names, labels)",
            "label_npy": "int64 labels aligned with data rows",
        },
        "class_map": DSG_CLASS_MAP,
        "splits": [asdict(summary) for summary in summaries],
    }
    write_json(output_dir / "metadata.json", metadata, overwrite=overwrite)
    LOGGER.info("dsg: converted %d protocol splits", len(summaries))
    return metadata


def discover_materialized_protocols(
    source_dir: Path,
) -> dict[str, dict[SplitName, list[DsgSourceSample]]] | None:
    split_dirs = {
        protocol: {
            split: source_dir / protocol / split
            for split in ("train", "test")
        }
        for protocol in ("xsub", "xview")
    }
    existing = [
        directory
        for splits in split_dirs.values()
        for directory in splits.values()
        if directory.is_dir()
    ]
    if not existing:
        return None
    if len(existing) != 4:
        missing = [
            str(directory)
            for splits in split_dirs.values()
            for directory in splits.values()
            if not directory.is_dir()
        ]
        raise ValueError(
            "Incomplete materialized DSG layout; missing directories: "
            + ", ".join(missing)
        )

    protocols: dict[str, dict[SplitName, list[DsgSourceSample]]] = {}
    for protocol, splits in split_dirs.items():
        protocol_splits: dict[SplitName, list[DsgSourceSample]] = {}
        for split, directory in splits.items():
            split_name: SplitName = split  # type: ignore[assignment]
            paths = sorted(directory.glob("*.skeleton"))
            if not paths:
                raise ValueError(f"No .skeleton files found in {directory}")
            samples: list[DsgSourceSample] = []
            for path in paths:
                sample = parse_dsg_filename(path)
                if sample is None:
                    raise ValueError(
                        f"Unexpected DSG action or filename in {directory}: {path.name}"
                    )
                samples.append(sample)
            protocol_splits[split_name] = samples
        protocols[protocol] = protocol_splits
    return protocols


def find_dsg_raw_dir(source_dir: Path) -> Path:
    candidates = (source_dir, source_dir / "nturgb+d_skeletons")
    populated = [candidate for candidate in candidates if any(candidate.glob("*.skeleton"))]
    if len(populated) == 1:
        return populated[0]
    if len(populated) > 1:
        raise ValueError(
            "DSG skeleton files were found in more than one supported location: "
            + ", ".join(str(path) for path in populated)
        )

    raise FileNotFoundError(
        f"No .skeleton files found in {source_dir} or "
        f"{source_dir / 'nturgb+d_skeletons'}"
    )


def discover_dsg_samples(raw_dir: Path) -> list[DsgSourceSample]:
    samples: list[DsgSourceSample] = []
    for path in sorted(raw_dir.glob("*.skeleton")):
        parsed = parse_dsg_filename(path)
        if parsed is not None:
            samples.append(parsed)
    if not samples:
        raise ValueError(
            f"No selected DSG actions ({', '.join(DSG_ACTION_LABELS)}) found in {raw_dir}"
        )
    return samples


def parse_dsg_filename(path: Path) -> DsgSourceSample | None:
    match = DSG_FILENAME_PATTERN.fullmatch(path.name)
    if match is None:
        return None
    action_id = match.group("action").upper()
    if action_id not in DSG_ACTION_LABELS:
        return None
    return DsgSourceSample(
        path=path,
        action_id=action_id,
        label=DSG_ACTION_LABELS[action_id],
        subject=int(match.group("subject")),
        camera=int(match.group("camera")),
    )


def build_official_protocols(
    samples: Sequence[DsgSourceSample],
) -> dict[str, dict[SplitName, list[DsgSourceSample]]]:
    return {
        "xsub": {
            "train": [
                sample for sample in samples if sample.subject in DSG_TRAINING_SUBJECTS
            ],
            "test": [
                sample
                for sample in samples
                if sample.subject not in DSG_TRAINING_SUBJECTS
            ],
        },
        "xview": {
            "train": [
                sample for sample in samples if sample.camera in DSG_TRAINING_CAMERAS
            ],
            "test": [
                sample
                for sample in samples
                if sample.camera not in DSG_TRAINING_CAMERAS
            ],
        },
    }


def validate_official_protocols(
    protocols: dict[str, dict[SplitName, list[DsgSourceSample]]],
) -> None:
    errors: list[str] = []
    protocol_names: dict[str, set[str]] = {}
    for protocol, expected_splits in DSG_EXPECTED_SPLIT_COUNTS.items():
        train_names = {
            sample.path.name for sample in protocols[protocol]["train"]
        }
        test_names = {sample.path.name for sample in protocols[protocol]["test"]}
        if len(train_names) != len(protocols[protocol]["train"]):
            errors.append(f"{protocol}/train: duplicate sample names")
        if len(test_names) != len(protocols[protocol]["test"]):
            errors.append(f"{protocol}/test: duplicate sample names")
        overlap = train_names & test_names
        if overlap:
            errors.append(f"{protocol}: {len(overlap)} train/test samples overlap")
        protocol_names[protocol] = train_names | test_names

        for split, expected_count in expected_splits.items():
            split_name: SplitName = split  # type: ignore[assignment]
            samples = protocols[protocol][split_name]
            labels = {sample.label for sample in samples}
            if len(samples) != expected_count:
                errors.append(
                    f"{protocol}/{split}: expected {expected_count}, got {len(samples)}"
                )
            if labels != set(DSG_ACTION_LABELS.values()):
                errors.append(
                    f"{protocol}/{split}: expected labels 0-4, got {sorted(labels)}"
                )
            for sample in samples:
                expected_split = (
                    "train"
                    if (
                        sample.subject in DSG_TRAINING_SUBJECTS
                        if protocol == "xsub"
                        else sample.camera in DSG_TRAINING_CAMERAS
                    )
                    else "test"
                )
                if split != expected_split:
                    errors.append(
                        f"{protocol}/{split}: {sample.path.name} belongs in "
                        f"{expected_split}"
                    )
                    break
    if protocol_names.get("xsub") != protocol_names.get("xview"):
        errors.append("xsub and xview do not contain the same source sample names")
    if errors:
        raise ValueError(
            "DSG source does not contain the complete five-class NTU RGB+D 60 "
            "protocol data:\n  " + "\n  ".join(errors)
        )


def write_dsg_array(
    skeleton_paths: Sequence[Path],
    action_ids: Sequence[str],
    labels: Sequence[int],
    protocol: str,
    split: SplitName,
    source_dir: Path,
    destination: Path,
    overwrite: bool,
    show_progress: bool,
) -> list[DsgSampleRecord]:
    prepare_destination(destination, overwrite)
    tmp_path = destination.with_name(f".{destination.name}.tmp")
    if tmp_path.exists() or tmp_path.is_symlink():
        tmp_path.unlink()

    shape = (
        len(skeleton_paths),
        3,
        DSG_MAX_FRAMES,
        DSG_NUM_JOINTS,
        DSG_MAX_BODIES,
    )
    records: list[DsgSampleRecord] = []
    progress = ProgressBar(f"dsg {protocol}/{split}", len(skeleton_paths), show_progress)
    data: np.memmap | None = None
    try:
        data = np.lib.format.open_memmap(
            tmp_path,
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )
        for index, (path, action_id, label) in enumerate(
            zip(skeleton_paths, action_ids, labels, strict=True)
        ):
            sample, stats = parse_ntu_skeleton(path)
            if sample.shape[1] > 0:
                data[index, :, : sample.shape[1], :, :] = sample

            warnings: list[str] = []
            if stats["frame_count"] > DSG_MAX_FRAMES:
                warnings.append("frames_truncated")
            if stats["dropped_body_instances"] > 0:
                warnings.append("bodies_dropped")
            if stats["empty_frame_count"] > 0:
                warnings.append("empty_frames")

            records.append(
                DsgSampleRecord(
                    sample_name=path.name,
                    protocol=protocol,
                    split=split,
                    action_id=action_id,
                    label=label,
                    source_path=relative_to(path, source_dir),
                    frame_count=stats["frame_count"],
                    stored_frame_count=min(stats["frame_count"], DSG_MAX_FRAMES),
                    max_body_count=stats["max_body_count"],
                    empty_frame_count=stats["empty_frame_count"],
                    dropped_body_instances=stats["dropped_body_instances"],
                    warnings=warnings,
                )
            )
            progress.update(index + 1)

        data.flush()
        del data
        data = None
        tmp_path.replace(destination)
    except BaseException:
        if data is not None:
            del data
        if tmp_path.exists() or tmp_path.is_symlink():
            tmp_path.unlink()
        raise
    return records


def parse_ntu_skeleton(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    line_number = [0]
    with path.open("r", encoding="utf-8") as file:
        frame_count = parse_integer_line(
            read_required_line(file, path, line_number, "frame count"),
            path,
            line_number[0],
            "frame count",
        )
        if frame_count < 0:
            raise ValueError(f"{path}:{line_number[0]} frame count cannot be negative")

        stored_frames = min(frame_count, DSG_MAX_FRAMES)
        sample = np.zeros(
            (3, stored_frames, DSG_NUM_JOINTS, DSG_MAX_BODIES),
            dtype=np.float32,
        )
        max_body_count = 0
        empty_frame_count = 0
        dropped_body_instances = 0

        for frame_index in range(frame_count):
            body_count = parse_integer_line(
                read_required_line(file, path, line_number, "body count"),
                path,
                line_number[0],
                "body count",
            )
            if body_count < 0:
                raise ValueError(f"{path}:{line_number[0]} body count cannot be negative")
            max_body_count = max(max_body_count, body_count)
            if body_count == 0:
                empty_frame_count += 1
            dropped_body_instances += max(0, body_count - DSG_MAX_BODIES)

            for body_index in range(body_count):
                body_info = read_required_line(file, path, line_number, "body info").split()
                if len(body_info) < 10:
                    raise ValueError(
                        f"{path}:{line_number[0]} body info must contain at least 10 values"
                    )
                joint_count = parse_integer_line(
                    read_required_line(file, path, line_number, "joint count"),
                    path,
                    line_number[0],
                    "joint count",
                )
                if joint_count != DSG_NUM_JOINTS:
                    raise ValueError(
                        f"{path}:{line_number[0]} expected {DSG_NUM_JOINTS} joints, "
                        f"got {joint_count}"
                    )

                keep_body = (
                    frame_index < stored_frames and body_index < DSG_MAX_BODIES
                )
                for joint_index in range(joint_count):
                    values = read_required_line(
                        file,
                        path,
                        line_number,
                        "joint info",
                    ).split()
                    if len(values) < 12:
                        raise ValueError(
                            f"{path}:{line_number[0]} joint info must contain at least 12 values"
                        )
                    if keep_body:
                        try:
                            sample[:, frame_index, joint_index, body_index] = (
                                float(values[0]),
                                float(values[1]),
                                float(values[2]),
                            )
                        except ValueError as exc:
                            raise ValueError(
                                f"{path}:{line_number[0]} invalid x/y/z joint coordinates"
                            ) from exc

        if file.read().strip():
            raise ValueError(f"{path}: unexpected data after {frame_count} frames")

    return sample, {
        "frame_count": frame_count,
        "max_body_count": max_body_count,
        "empty_frame_count": empty_frame_count,
        "dropped_body_instances": dropped_body_instances,
    }


def read_required_line(
    file: Any,
    path: Path,
    line_number: list[int],
    field: str,
) -> str:
    line = file.readline()
    line_number[0] += 1
    if line == "":
        raise ValueError(f"{path}:{line_number[0]} unexpected EOF while reading {field}")
    return line.strip()


def parse_integer_line(value: str, path: Path, line_number: int, field: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number} invalid {field}: {value!r}") from exc
