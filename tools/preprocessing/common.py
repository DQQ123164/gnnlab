"""Shared types, file operations, progress reporting, and cleanup."""

from __future__ import annotations

import json
import logging
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np


LOGGER = logging.getLogger("gnnlab-preprocess")

DatasetName = Literal["dsg", "ssg"]
SplitName = Literal["train", "test"]


def clean_processed_datasets(
    output_dir: Path,
    selected: set[DatasetName],
) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Processed output is not a directory: {output_dir}")

    removed = 0
    for dataset in sorted(selected):
        target = output_dir / dataset
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        else:
            LOGGER.info("Nothing to clean for %s: %s", dataset, target)
            continue
        removed += 1
        LOGGER.info("Removed generated %s data: %s", dataset, target)

    if removed == 0:
        LOGGER.info("No generated dataset output was found under %s", output_dir)


class ProgressBar:
    def __init__(self, label: str, total: int, enabled: bool) -> None:
        self.label = label
        self.total = total
        self.enabled = enabled
        self.started_at = time.monotonic()
        self.last_rendered_at = 0.0
        self.log_interval = max(1, total // 20)

    def update(self, completed: int) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        elapsed = max(now - self.started_at, 1e-9)
        rate = completed / elapsed
        remaining = max(self.total - completed, 0)
        eta = remaining / rate if rate > 0 else 0.0
        percent = completed / self.total if self.total else 1.0

        if sys.stderr.isatty():
            if completed < self.total and now - self.last_rendered_at < 0.1:
                return
            width = 28
            filled = min(width, int(width * percent))
            bar = "#" * filled + "-" * (width - filled)
            message = (
                f"\r{self.label:<20} [{bar}] {completed:>{len(str(self.total))}}/"
                f"{self.total} {percent:6.2%} {rate:6.1f} samples/s "
                f"ETA {format_duration(eta)}"
            )
            print(
                message,
                file=sys.stderr,
                end="\n" if completed == self.total else "",
                flush=True,
            )
            self.last_rendered_at = now
        elif completed == self.total or completed % self.log_interval == 0:
            LOGGER.info(
                "%s progress: %d/%d (%.1f%%), %.1f samples/s, ETA %s",
                self.label,
                completed,
                self.total,
                percent * 100,
                rate,
                format_duration(eta),
            )


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def atomic_save_npy(path: Path, array: np.ndarray, overwrite: bool) -> None:
    prepare_destination(path, overwrite)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("wb") as file:
        np.save(file, array)
    tmp_path.replace(path)


def write_pickle(path: Path, payload: Any, overwrite: bool) -> None:
    prepare_destination(path, overwrite)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def write_json(path: Path, payload: Any, overwrite: bool) -> None:
    prepare_destination(path, overwrite)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    tmp_path.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], overwrite: bool) -> None:
    prepare_destination(path, overwrite)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        for row in rows:
            json.dump(row, file, sort_keys=True)
            file.write("\n")
    tmp_path.replace(path)


def write_text(path: Path, payload: str, overwrite: bool) -> None:
    prepare_destination(path, overwrite)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        file.write(payload)
    tmp_path.replace(path)


def prepare_destination(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if (path.exists() or path.is_symlink()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory not found: {path}")


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def relative_to(path: Path, root: Path) -> str:
    return str(path.absolute().relative_to(root.absolute()))


def count_labels(labels: Sequence[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        key = str(int(label))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def count_warning_types(
    records: Sequence[Any],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for warning in record.warnings:
            counts[warning] = counts.get(warning, 0) + 1
    return dict(sorted(counts.items()))
