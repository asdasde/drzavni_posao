from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

import numpy as np
from numpy.typing import NDArray

from .models import SegmentRecord


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} on line {line_number}: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} on line {line_number}"
                )
            yield value


def load_segments(path: Path) -> list[SegmentRecord]:
    """Load and validate the conformed segment corpus."""
    segments = [SegmentRecord.model_validate(row) for row in read_jsonl(path)]
    if not segments:
        raise ValueError(f"No segments found in {path}")
    ids = [segment.segment_id for segment in segments]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate segment IDs in {path}")
    return segments


@contextmanager
def atomic_writer(path: Path, *, binary: bool = False) -> Iterator[IO[Any]]:
    """Write via a temp file, then atomically rename it into place.

    A crash mid-write leaves the temp file (cleaned up here), never a truncated
    destination. Pass ``binary=True`` for bytes (e.g. ``np.save``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(
            descriptor, "wb" if binary else "w", encoding=None if binary else "utf-8"
        ) as handle:
            yield handle
        os.replace(temporary_name, path)
        os.chmod(path, 0o644)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with atomic_writer(path) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_json(path: Path, value: Any) -> None:
    with atomic_writer(path) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_float32_matrix(path: Path, matrix: NDArray[np.float32]) -> None:
    """Write a 2-D matrix as raw C-order float32 bytes, atomically.

    The layout is not self-describing: the caller records the shape and
    ``sys.byteorder`` in its own manifest and passes them back to
    :func:`read_float32_matrix`.
    """
    with atomic_writer(path, binary=True) as handle:
        np.ascontiguousarray(matrix, dtype=np.float32).tofile(handle)


def read_float32_matrix(
    path: Path, *, rows: int, cols: int, byte_order: str
) -> NDArray[np.float32]:
    """Read a matrix written by :func:`write_float32_matrix`.

    ``byte_order`` is the writer's ``sys.byteorder`` from the manifest; the data
    is byteswapped when it differs from this machine's.
    """
    expected = rows * cols
    values = np.fromfile(path, dtype=np.float32, count=expected)
    if values.size != expected:
        raise ValueError(f"Float32 matrix at {path} is truncated")
    if byte_order != sys.byteorder:
        values.byteswap(inplace=True)
    return values.reshape(rows, cols)
