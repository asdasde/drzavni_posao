#!/usr/bin/env python3
"""Render QLoRA holdout comparisons by episode key.

Example:

    python scripts/render_qlora_holdouts.py \
      artifacts/qlora_holdout/<run_id>/heldout-comparisons
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from drzavni_posao.transcription.episode import episode_order_key


TURN_RE = re.compile(r"\[([^\]]+)\]\s*")
TEXT_FIELDS = ("reference", "base", "qlora")


def as_lines(text: str | None) -> str:
    """Render one character segment per line as ``[Speaker] text``."""
    parts = TURN_RE.split(text or "")
    lines = []
    lead = parts[0].strip()
    if lead:
        lines.append(" ".join(lead.split()))
    for speaker, body in zip(parts[1::2], parts[2::2]):
        body = " ".join(body.split())
        lines.append(f"[{speaker}] {body}".strip())
    return "\n".join(lines)


def load_by_episode(directory: Path) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        episode_key = item.get("episode_key")
        if not isinstance(episode_key, str) or not episode_key:
            raise ValueError(f"{path.name} has no valid episode_key")
        if episode_key in items:
            raise ValueError(f"Duplicate episode_key {episode_key!r}")
        items[episode_key] = {
            **item,
            **{field: as_lines(item.get(field)) for field in TEXT_FIELDS},
            "source_artifact": path.name,
        }
    if not items:
        raise ValueError(f"No comparison JSON files found in {directory}")
    return items


def render_episode(episode_key: str, item: dict[str, Any]) -> str:
    sections = [
        f"# Episode {episode_key}",
        "",
        f"- Source artifact: `{item['source_artifact']}`",
        f"- Holdout index: {item.get('holdout_index', 'unknown')}",
        "",
        "## Premise",
        "",
        item.get("premise", ""),
    ]
    for field in TEXT_FIELDS:
        text = item.get(field)
        if text:
            sections.extend(
                [
                    "",
                    f"## {field.title()}",
                    "",
                    "````text",
                    text,
                    "````",
                ]
            )
    return "\n".join(sections).rstrip() + "\n"


def render_all(input_dir: Path, output_dir: Path) -> Path:
    items = load_by_episode(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_lines = [
        "# QLoRA holdout comparisons by episode",
        "",
        f"Rendered from `{input_dir}`.",
        "",
    ]
    for episode_key, item in sorted(
        items.items(),
        key=lambda item: episode_order_key(
            int(item[0]) if item[0].isdigit() else None, item[0]
        ),
    ):
        filename = _episode_filename(episode_key)
        (output_dir / filename).write_text(
            render_episode(episode_key, item),
            encoding="utf-8",
        )
        index_lines.append(
            f"- [Episode {episode_key}]({filename}) — {item.get('premise', '')}"
        )

    (output_dir / "index.md").write_text(
        "\n".join(index_lines).rstrip() + "\n",
        encoding="utf-8",
    )
    (output_dir / "by_episode_key.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir


def _episode_filename(episode_key: str) -> str:
    slug = re.sub(r"[^\w.-]+", "-", episode_key, flags=re.UNICODE).strip("-.")
    if not slug:
        raise ValueError(f"Cannot create a filename for episode {episode_key!r}")
    return f"episode-{slug}.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing heldout-comparisons/*.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: <input parent>/rendered)",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir.parent / "rendered"
    print(f"Wrote {render_all(args.input_dir, output_dir)}")


if __name__ == "__main__":
    main()
