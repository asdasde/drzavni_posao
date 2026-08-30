from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

from .models import ProcessingRecord, ProcessingStatus


class JsonlStateStore:
    """Append-only state; every completed per-post attempt is immediately durable."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def append(self, record: ProcessingRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def records(self) -> Iterator[ProcessingRecord]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield ProcessingRecord.model_validate_json(line)
                except (ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"Invalid collector state in {self.path} on line {line_number}"
                    ) from error

    def latest_by_post(self) -> dict[str, ProcessingRecord]:
        latest: dict[str, ProcessingRecord] = {}
        for record in self.records():
            # A duplicate observation is an audit event, not the post's result.
            if record.status is not ProcessingStatus.DUPLICATE:
                latest[record.post_id] = record
        return latest

    def completed_post_ids(self) -> set[str]:
        return set(self.latest_by_post())

    def accepted_records(self) -> Iterator[ProcessingRecord]:
        """Yield latest accepted rows for an eval-specific exporter."""
        for record in self.latest_by_post().values():
            if record.status is ProcessingStatus.ACCEPTED:
                yield record
