#!/usr/bin/env python3
"""Recover the original five-class DSG source from an NTU60 annotation file."""

from __future__ import annotations

import argparse
import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__:
    from .preprocessing.config import env_path, load_env_file, optional_env_path
    from .preprocessing.common import LOGGER, ProgressBar
    from .preprocessing.dsg import (
        DSG_ACTION_LABELS,
        build_official_protocols,
        discover_dsg_samples,
        validate_official_protocols,
    )
else:
    from preprocessing.config import env_path, load_env_file, optional_env_path
    from preprocessing.common import LOGGER, ProgressBar
    from preprocessing.dsg import (
        DSG_ACTION_LABELS,
        build_official_protocols,
        discover_dsg_samples,
        validate_official_protocols,
    )


EXPECTED_ACTION_COUNTS = {
    "A059": 939,
    "A030": 944,
    "A016": 940,
    "A005": 942,
    "A027": 948,
}
BODY_INFO = "0 0 0 0 0 0 0 0 0 0\n"
JOINT_EXTRA = " 0 0 0 0 0 0 0 0 2\n"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    load_env_file(project_root / ".env")
    dsg_dir = env_path("GNNLAB_DSG_DIR", project_root / "dsg")

    parser = argparse.ArgumentParser(
        description="Export the original five DSG actions as NTU .skeleton files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=optional_env_path("NTU60_ANNOTATION_FILE"),
        help=(
            "Path to the NTU60 ntu60_3danno.pkl file. Can also be set with "
            "NTU60_ANNOTATION_FILE."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=dsg_dir / "nturgb+d_skeletons",
        help="Directory receiving the 4713 selected .skeleton files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace .skeleton files already present in the output directory.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress reporting.",
    )
    args = parser.parse_args()
    if args.annotation is None:
        parser.error("--annotation or NTU60_ANNOTATION_FILE is required")
    return args


def load_selected_annotations(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"NTU60 annotation file not found: {path}")
    LOGGER.info("Loading NTU60 annotations from %s", path)
    with path.open("rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict) or not isinstance(payload.get("annotations"), list):
        raise ValueError(f"Unexpected NTU60 annotation structure: {path}")

    source_labels = {int(action_id[1:]) - 1 for action_id in DSG_ACTION_LABELS}
    selected = [
        annotation
        for annotation in payload["annotations"]
        if int(annotation.get("label", -1)) in source_labels
    ]
    counts = Counter(f"A{int(annotation['label']) + 1:03d}" for annotation in selected)
    if dict(counts) != EXPECTED_ACTION_COUNTS:
        raise ValueError(
            "Unexpected DSG class counts in NTU60 annotation: "
            f"expected {EXPECTED_ACTION_COUNTS}, got {dict(counts)}"
        )
    return selected


def prepare_output(directory: Path, overwrite: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    existing = list(directory.glob("*.skeleton"))
    if existing and not overwrite:
        raise FileExistsError(
            f"Found {len(existing)} existing .skeleton files in {directory}; "
            "use --overwrite to replace them"
        )
    if overwrite:
        for path in existing:
            path.unlink()


def export_annotations(
    annotations: Sequence[dict[str, Any]],
    output_dir: Path,
    show_progress: bool,
) -> None:
    progress = ProgressBar("export dsg source", len(annotations), show_progress)
    for index, annotation in enumerate(annotations):
        sample_name = str(annotation.get("frame_dir", ""))
        if not sample_name or Path(sample_name).name != sample_name:
            raise ValueError(f"Invalid NTU sample name: {sample_name!r}")

        action_id = f"A{int(annotation['label']) + 1:03d}"
        if not sample_name.upper().endswith(action_id):
            raise ValueError(
                f"Annotation label {action_id} does not match sample name {sample_name}"
            )

        keypoint = np.asarray(annotation.get("keypoint"), dtype=np.float32)
        if keypoint.ndim != 4 or keypoint.shape[2:] != (25, 3):
            raise ValueError(
                f"Unexpected keypoint shape for {sample_name}: {keypoint.shape}"
            )
        write_skeleton(output_dir / f"{sample_name}.skeleton", keypoint)
        progress.update(index + 1)


def write_skeleton(path: Path, keypoint: np.ndarray) -> None:
    people, frames, joints, _channels = keypoint.shape
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="ascii") as file:
            file.write(f"{frames}\n")
            for frame in range(frames):
                file.write(f"{people}\n")
                for person in range(people):
                    file.write(BODY_INFO)
                    file.write(f"{joints}\n")
                    for x, y, z in keypoint[person, frame, :, :3]:
                        file.write(
                            f"{float(x):.9g} {float(y):.9g} {float(z):.9g}"
                            f"{JOINT_EXTRA}"
                        )
        temporary.replace(path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    annotations = load_selected_annotations(args.annotation.resolve())
    output_dir = args.output_dir.resolve()
    prepare_output(output_dir, args.overwrite)
    export_annotations(annotations, output_dir, not args.no_progress)

    samples = discover_dsg_samples(output_dir)
    validate_official_protocols(build_official_protocols(samples))
    LOGGER.info("Exported and validated %d DSG skeleton files under %s", len(samples), output_dir)


if __name__ == "__main__":
    main()
