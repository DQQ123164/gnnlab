#!/usr/bin/env python3
"""Build the five-class DSG source splits directly from the NTU60 archive."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import stat
import tempfile
import threading
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

if __package__:
    from .preprocessing.common import LOGGER, ProgressBar
    from .preprocessing.config import env_path, load_env_file
    from .preprocessing.dsg import (
        DSG_EXCLUDED_SAMPLE_NAMES,
        DSG_EXPECTED_ACTION_COUNTS,
        DsgSourceSample,
        build_official_protocols,
        discover_materialized_protocols,
        parse_dsg_filename,
        validate_official_protocols,
    )
else:
    from preprocessing.common import LOGGER, ProgressBar
    from preprocessing.config import env_path, load_env_file
    from preprocessing.dsg import (
        DSG_EXCLUDED_SAMPLE_NAMES,
        DSG_EXPECTED_ACTION_COUNTS,
        DsgSourceSample,
        build_official_protocols,
        discover_materialized_protocols,
        parse_dsg_filename,
        validate_official_protocols,
    )


ARCHIVE_NAME = "nturgbd_skeletons_s001_to_s017.zip"
MANIFEST_NAME = ".dsg-split-manifest.json"
PROTOCOLS = ("xsub", "xview")
SPLITS = ("train", "test")
DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))


@dataclass(frozen=True)
class ArchiveSample:
    info: zipfile.ZipInfo
    sample: DsgSourceSample


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    load_env_file(project_root / ".env")
    dsg_dir = env_path("GNNLAB_DSG_DIR", project_root / "dsg")

    parser = argparse.ArgumentParser(
        description=(
            "Split the five DSG actions from the original NTU60 skeleton ZIP "
            "into official XSub and XView directories."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help=f"Directory containing {ARCHIVE_NAME}.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=dsg_dir,
        help="Directory receiving xsub/, xview/, and the split manifest.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing DSG split directories after staging validation.",
    )
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy each sample across protocols instead of using a hard link.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate archive inventory and official splits without writing files.",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=DEFAULT_WORKERS,
        help="Number of worker threads used to extract selected samples.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress reporting.",
    )
    return parser.parse_args()


def validate_archive_member(info: zipfile.ZipInfo) -> str | None:
    name = info.orig_filename
    if not name or "\x00" in name or "\\" in name:
        raise ValueError(f"Unsafe ZIP member name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe ZIP member path: {name!r}")
    file_type = (info.external_attr >> 16) & 0o170000
    if file_type == stat.S_IFLNK:
        raise ValueError(f"Symbolic-link ZIP member is not allowed: {name}")
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ValueError(f"Unsupported special-file ZIP member: {name}")
    if info.flag_bits & 0x1:
        raise ValueError(f"Encrypted ZIP member is not allowed: {name}")
    if info.is_dir():
        return None
    basename = path.name
    if not basename or basename != Path(basename).name:
        raise ValueError(f"Invalid ZIP member basename: {name!r}")
    return basename


def scan_archive(archive: zipfile.ZipFile) -> tuple[list[ArchiveSample], int]:
    seen_basenames: set[str] = set()
    valid: list[ArchiveSample] = []
    excluded_found: set[str] = set()

    for info in archive.infolist():
        basename = validate_archive_member(info)
        if basename is None:
            continue
        if basename in seen_basenames:
            raise ValueError(f"Duplicate ZIP member basename: {basename}")
        seen_basenames.add(basename)

        parsed = parse_dsg_filename(Path(basename))
        if parsed is None:
            continue
        if basename in DSG_EXCLUDED_SAMPLE_NAMES:
            excluded_found.add(basename)
            continue
        valid.append(ArchiveSample(info=info, sample=parsed))

    missing_exclusions = DSG_EXCLUDED_SAMPLE_NAMES - excluded_found
    if missing_exclusions:
        raise ValueError(
            "Archive is missing known invalid NTU60 samples: "
            + ", ".join(sorted(missing_exclusions))
        )
    if len(excluded_found) != len(DSG_EXCLUDED_SAMPLE_NAMES):
        raise ValueError("Unexpected invalid-sample count in NTU60 archive")

    names = [row.sample.path.name for row in valid]
    if len(names) != len(set(names)):
        raise ValueError("Selected DSG samples contain duplicate basenames")
    action_counts = Counter(row.sample.action_id for row in valid)
    if dict(action_counts) != DSG_EXPECTED_ACTION_COUNTS:
        raise ValueError(
            "Unexpected DSG class counts in NTU60 archive: "
            f"expected {DSG_EXPECTED_ACTION_COUNTS}, got {dict(action_counts)}"
        )
    return sorted(valid, key=lambda row: row.sample.path.name), len(excluded_found)


def protocol_split_counts(
    protocols: dict[str, dict[str, list[DsgSourceSample]]],
) -> dict[str, dict[str, int]]:
    return {
        protocol: {split: len(rows) for split, rows in splits.items()}
        for protocol, splits in protocols.items()
    }


def make_manifest(
    archive_path: Path,
    samples: Sequence[ArchiveSample],
    excluded_count: int,
    protocols: dict[str, dict[str, list[DsgSourceSample]]],
    copy_files: bool,
    workers: int,
) -> dict[str, object]:
    return {
        "archive": str(archive_path),
        "valid_samples": len(samples),
        "excluded_samples": excluded_count,
        "class_counts": dict(Counter(row.sample.action_id for row in samples)),
        "protocol_splits": protocol_split_counts(protocols),
        "link_mode": "copy" if copy_files else "hardlink",
        "workers": workers,
    }


def destination_map(
    protocols: dict[str, dict[str, list[DsgSourceSample]]],
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for protocol, splits in protocols.items():
        for split, samples in splits.items():
            for sample in samples:
                result.setdefault(sample.path.name, {})[protocol] = split
    return result


def materialize_staging(
    archive_path: Path,
    samples: Sequence[ArchiveSample],
    protocols: dict[str, dict[str, list[DsgSourceSample]]],
    staging: Path,
    manifest: dict[str, object],
    copy_files: bool,
    show_progress: bool,
    workers: int,
) -> None:
    locations = destination_map(protocols)
    for protocol in PROTOCOLS:
        for split in SPLITS:
            (staging / protocol / split).mkdir(parents=True, exist_ok=True)

    thread_state = threading.local()
    worker_archives: list[zipfile.ZipFile] = []
    worker_archives_lock = threading.Lock()

    def worker_archive() -> zipfile.ZipFile:
        archive = getattr(thread_state, "archive", None)
        if archive is None:
            archive = zipfile.ZipFile(archive_path, "r")
            thread_state.archive = archive
            with worker_archives_lock:
                worker_archives.append(archive)
        return archive

    def extract_sample(row: ArchiveSample) -> None:
        basename = row.sample.path.name
        split_by_protocol = locations[basename]
        first = staging / "xsub" / split_by_protocol["xsub"] / basename
        second = staging / "xview" / split_by_protocol["xview"] / basename
        archive = worker_archive()
        with archive.open(row.info, "r") as source, first.open("xb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        if copy_files:
            shutil.copyfile(first, second)
        else:
            os.link(first, second)

    progress = ProgressBar("split dsg archive", len(samples), show_progress)
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(extract_sample, row) for row in samples]
            try:
                for completed, future in enumerate(as_completed(futures), start=1):
                    future.result()
                    progress.update(completed)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    finally:
        for archive in worker_archives:
            archive.close()

    manifest_path = staging / MANIFEST_NAME
    with manifest_path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")

    materialized = discover_materialized_protocols(staging)
    if materialized is None:
        raise RuntimeError("Staged DSG validation unexpectedly found no splits")
    validate_official_protocols(materialized)


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def publish_staging(staging: Path, output_root: Path, overwrite: bool) -> None:
    target_names = (*PROTOCOLS, MANIFEST_NAME)
    targets = [output_root / name for name in target_names]
    existing = [path for path in targets if path.exists() or path.is_symlink()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to replace existing DSG outputs: "
            + ", ".join(str(path) for path in existing)
            + "; use --overwrite to replace them"
        )

    backup = Path(
        tempfile.mkdtemp(prefix=".dsg-split-backup-", dir=output_root)
    )
    moved_new: list[Path] = []
    try:
        for target in existing:
            target.replace(backup / target.name)
        for name in target_names:
            target = output_root / name
            (staging / name).replace(target)
            moved_new.append(target)
    except BaseException as publish_error:
        rollback_errors: list[str] = []
        for target in reversed(moved_new):
            try:
                remove_path(target)
            except BaseException as exc:
                rollback_errors.append(f"could not remove {target}: {exc}")
        try:
            saved_paths = list(backup.iterdir())
        except BaseException as exc:
            rollback_errors.append(f"could not inspect backup directory: {exc}")
            saved_paths = []
        for saved in saved_paths:
            try:
                target = output_root / saved.name
                if target.exists() or target.is_symlink():
                    rollback_errors.append(
                        f"could not restore {saved.name}: target still exists at {target}"
                    )
                    continue
                saved.replace(target)
            except BaseException as exc:
                rollback_errors.append(f"could not restore {saved.name}: {exc}")

        if rollback_errors:
            details = "; ".join(rollback_errors)
            raise RuntimeError(
                "DSG publish failed and rollback was incomplete. "
                f"Backup data was retained at {backup}: {details}"
            ) from publish_error
        remove_path(backup)
        raise
    else:
        remove_path(backup)


def split_archive(
    source_dir: Path,
    output_root: Path,
    overwrite: bool,
    copy_files: bool,
    dry_run: bool,
    show_progress: bool,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be a positive integer")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"NTU60 source directory not found: {source_dir}")
    archive_path = source_dir / ARCHIVE_NAME
    if not archive_path.is_file():
        raise FileNotFoundError(f"Required NTU60 archive not found: {archive_path}")

    output_root = output_root.expanduser().absolute()
    if output_root.is_symlink():
        raise ValueError(f"--output-root must not be a symbolic link: {output_root}")
    output_root = output_root.resolve()
    source_dir = source_dir.expanduser().resolve()
    archive_path = archive_path.resolve()
    if output_root == source_dir or output_root in archive_path.parents:
        raise ValueError("--output-root must not contain or equal the NTU60 source directory")
    if output_root == Path(output_root.anchor):
        raise ValueError("Refusing to use a filesystem root as --output-root")
    with zipfile.ZipFile(archive_path, "r") as archive:
        samples, excluded_count = scan_archive(archive)
        source_samples = [row.sample for row in samples]
        protocols = build_official_protocols(source_samples)
        validate_official_protocols(protocols)
        manifest = make_manifest(
            archive_path,
            samples,
            excluded_count,
            protocols,
            copy_files,
            workers,
        )
        if dry_run:
            return manifest

        target_paths = [output_root / name for name in (*PROTOCOLS, MANIFEST_NAME)]
        existing = [path for path in target_paths if path.exists() or path.is_symlink()]
        if existing and not overwrite:
            raise FileExistsError(
                "Refusing to replace existing DSG outputs: "
                + ", ".join(str(path) for path in existing)
                + "; use --overwrite to replace them"
            )
        output_root.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=".dsg-split-staging-", dir=output_root)
        )
        try:
            materialize_staging(
                archive_path,
                samples,
                protocols,
                staging,
                manifest,
                copy_files,
                show_progress,
                workers,
            )
            publish_staging(staging, output_root, overwrite)
        finally:
            remove_path(staging)
    return manifest


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        manifest = split_archive(
            source_dir=args.source_dir,
            output_root=args.output_root,
            overwrite=args.overwrite,
            copy_files=args.copy_files,
            dry_run=args.dry_run,
            show_progress=not args.no_progress,
            workers=args.workers,
        )
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.dry_run:
        LOGGER.info("Dry run completed; no files were written")
    else:
        LOGGER.info("Published official DSG splits under %s", args.output_root.resolve())


if __name__ == "__main__":
    main()
