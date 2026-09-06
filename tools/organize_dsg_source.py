#!/usr/bin/env python3
"""Materialize official DSG XSub/XView train/test source directories."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

if __package__:
    from .preprocessing.common import LOGGER, ProgressBar
    from .preprocessing.dsg import (
        build_official_protocols,
        discover_dsg_samples,
        discover_materialized_protocols,
        validate_official_protocols,
    )
else:
    from preprocessing.common import LOGGER, ProgressBar
    from preprocessing.dsg import (
        build_official_protocols,
        discover_dsg_samples,
        discover_materialized_protocols,
        validate_official_protocols,
    )


PROTOCOLS = ("xsub", "xview")
SPLITS = ("train", "test")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Organize raw DSG skeletons into official protocol directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=root / "dsg" / "nturgb+d_skeletons",
        help="Directory containing all 4713 correct DSG .skeleton files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "dsg",
        help="Directory receiving xsub/ and xview/.",
    )
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy file contents instead of using space-saving hard links.",
    )
    parser.add_argument(
        "--remove-source",
        action="store_true",
        help="Remove SOURCE_DIR after all materialized splits pass validation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing xsub/ and xview/ directories after validation.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress reporting.",
    )
    return parser.parse_args()


def organize(
    source_dir: Path,
    output_root: Path,
    overwrite: bool,
    copy_files: bool,
    remove_source: bool,
    show_progress: bool,
) -> None:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"DSG source directory not found: {source_dir}")

    samples = discover_dsg_samples(source_dir)
    protocols = build_official_protocols(samples)
    validate_official_protocols(protocols)

    targets = [output_root / protocol for protocol in PROTOCOLS]
    existing = [target for target in targets if target.exists() or target.is_symlink()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to replace existing protocol directories: "
            + ", ".join(str(path) for path in existing)
        )

    staging = output_root / ".dsg-splits.tmp"
    if staging.exists() or staging.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Staging directory already exists: {staging}")
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        else:
            staging.unlink()

    total = sum(len(rows) for splits in protocols.values() for rows in splits.values())
    progress = ProgressBar("organize dsg", total, show_progress)
    completed = 0
    try:
        for protocol, splits in protocols.items():
            for split, rows in splits.items():
                destination = staging / protocol / split
                destination.mkdir(parents=True, exist_ok=True)
                for sample in rows:
                    target = destination / sample.path.name
                    if copy_files:
                        shutil.copy2(sample.path, target)
                    else:
                        os.link(sample.path, target)
                    completed += 1
                    progress.update(completed)

        materialized = discover_materialized_protocols(staging)
        if materialized is None:
            raise RuntimeError("Materialized DSG validation unexpectedly found no splits")
        validate_official_protocols(materialized)

        for target in targets:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            elif target.exists() or target.is_symlink():
                target.unlink()
        for protocol in PROTOCOLS:
            (staging / protocol).replace(output_root / protocol)
        staging.rmdir()

        if remove_source:
            shutil.rmtree(source_dir)
    except BaseException:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise

    mode = "copies" if copy_files else "hard links"
    LOGGER.info("Materialized official DSG splits as %s under %s", mode, output_root)
    if remove_source:
        LOGGER.info("Removed source directory after validation: %s", source_dir)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    organize(
        source_dir=args.source_dir.resolve(),
        output_root=args.output_root.resolve(),
        overwrite=args.overwrite,
        copy_files=args.copy_files,
        remove_source=args.remove_source,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
