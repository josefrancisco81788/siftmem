#!/usr/bin/env python3
"""Build markdown retrieval index from canonical Siftmem JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from siftmem.lib import (
    DEFAULT_JSONL_FILES,
    DEFAULT_MEMORY_DIR,
    IMPORTANCE_FLOORS,
    TOPIC_MAX_PER_TYPE,
    build_bm25_index,
    importance_floor,
    load_jsonl_records,
    log_event,
    normalize_text,
    resolve_superseded_entry_ids,
)

def _default_output_dir(memory_dir: Path) -> Path:
    return memory_dir / "siftmem_index"


@dataclass
class MemoryEntry:
    timestamp: str
    entry_type: str
    topic: str
    content: str
    importance: float
    source_file: str
    below_threshold: bool = False
    superseded: bool = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Siftmem markdown retrieval index.")
    parser.add_argument("--memory-dir", default=str(DEFAULT_MEMORY_DIR))
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Markdown index output directory (default: <memory-dir>/siftmem_index).",
    )
    parser.add_argument(
        "--importance-threshold",
        type=float,
        default=None,
        help="Legacy default threshold (deprecated; use type floors).",
    )
    parser.add_argument(
        "--facts-importance-threshold",
        type=float,
        default=None,
        help="Legacy facts threshold (deprecated; fact floor is 0.60).",
    )
    parser.add_argument("--max-entry-chars", type=int, default=420)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print gating summary without writing index files.",
    )
    parser.add_argument(
        "--rebuild-bm25",
        action="store_true",
        default=True,
        help="Rebuild BM25 sidecar index (default: true).",
    )
    parser.add_argument("--no-rebuild-bm25", action="store_false", dest="rebuild_bm25")
    return parser.parse_args()


def _timestamp_sort_value(value: str) -> float:
    text = (value or "").strip()
    if not text:
        return 0.0
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return 0.0


def _slug(text: str) -> str:
    lower = text.lower().strip()
    lower = re.sub(r"[^a-z0-9]+", "-", lower)
    lower = lower.strip("-")
    return lower or "untitled-topic"


def _iter_entries(memory_dir: Path, max_chars: int) -> Iterable[MemoryEntry]:
    records = load_jsonl_records(memory_dir, DEFAULT_JSONL_FILES)
    superseded_ids = resolve_superseded_entry_ids(records)
    for record in records:
        if record.entry_id in superseded_ids:
            continue
        yield MemoryEntry(
            timestamp=record.timestamp,
            entry_type=record.entry_type,
            topic=record.topic,
            content=normalize_text(record.content, max_chars=max_chars),
            importance=record.importance,
            source_file=record.entry_id,
        )


def _passes_floor(entry: MemoryEntry) -> bool:
    return entry.importance >= importance_floor(entry.entry_type)


def _select_index_entries(entries: list[MemoryEntry]) -> tuple[list[MemoryEntry], dict[str, Any]]:
    """Apply per-type floors, topic caps, and min-one-per-topic fallback."""
    by_topic: dict[str, list[MemoryEntry]] = {}
    for row in entries:
        by_topic.setdefault(row.topic, []).append(row)

    selected: list[MemoryEntry] = []
    report: dict[str, Any] = {"topics": {}}

    for topic, topic_entries in sorted(by_topic.items()):
        topic_entries.sort(
            key=lambda r: (r.importance, _timestamp_sort_value(r.timestamp)),
            reverse=True,
        )
        passing = [r for r in topic_entries if _passes_floor(r)]
        gated_out = len(topic_entries) - len(passing)
        topic_selected: list[MemoryEntry] = []

        by_type: dict[str, list[MemoryEntry]] = {}
        for row in passing:
            by_type.setdefault(row.entry_type, []).append(row)

        for entry_type, typed_rows in by_type.items():
            cap = TOPIC_MAX_PER_TYPE.get(entry_type, 10)
            topic_selected.extend(typed_rows[:cap])

        if not topic_selected and topic_entries:
            fallback = topic_entries[0]
            fallback = MemoryEntry(
                timestamp=fallback.timestamp,
                entry_type=fallback.entry_type,
                topic=fallback.topic,
                content=f"[below-threshold] {fallback.content}",
                importance=fallback.importance,
                source_file=fallback.source_file,
                below_threshold=True,
            )
            topic_selected = [fallback]

        selected.extend(topic_selected)
        report["topics"][topic] = {
            "seen": len(topic_entries),
            "indexed": len(topic_selected),
            "gated_out": gated_out,
            "used_fallback": bool(topic_selected and topic_selected[0].below_threshold),
        }

    return selected, report


def _write_topic_file(output_dir: Path, topic: str, entries: list[MemoryEntry]) -> str:
    slug = _slug(topic)
    filename = f"topic__{slug}.md"
    path = output_dir / filename
    lines = [f"# Topic: {topic}", "", "## Siftmem Retrieval Entries", ""]
    for idx, item in enumerate(entries, start=1):
        lines.append(f"### Entry {idx}")
        lines.append(f"- importance: {item.importance:.3f}")
        lines.append(f"- timestamp: {item.timestamp or 'unknown'}")
        lines.append(f"- type: {item.entry_type or 'fact'}")
        lines.append(f"- source: {item.source_file}")
        if item.below_threshold:
            lines.append("- index_note: below-threshold fallback")
        lines.append("")
        lines.append(item.content)
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return filename


def _write_manifest(
    output_dir: Path,
    indexed_entries: int,
    topic_count: int,
    files: list[str],
    gating_report: dict[str, Any],
) -> None:
    manifest_path = output_dir / "SIFTMEM_INDEX.md"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "importance_floors": IMPORTANCE_FLOORS,
        "topic_max_per_type": TOPIC_MAX_PER_TYPE,
        "indexed_entries": indexed_entries,
        "topic_count": topic_count,
        "files": files,
        "gating": gating_report,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    body = [
        "# Siftmem Index Manifest",
        "",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        f"- importance_floors: {json.dumps(IMPORTANCE_FLOORS)}",
        f"- topic_max_per_type: {json.dumps(TOPIC_MAX_PER_TYPE)}",
        f"- indexed_entries: {indexed_entries}",
        f"- topic_count: {topic_count}",
        f"- content_digest_sha256: {digest}",
        "",
        "## Topic Files",
    ]
    body.extend([f"- {name}" for name in files])
    manifest_path.write_text("\n".join(body).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    memory_dir = Path(args.memory_dir)
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(memory_dir)

    entries = list(_iter_entries(memory_dir, max_chars=max(80, args.max_entry_chars)))
    selected, gating_report = _select_index_entries(entries)

    summary = {
        "memory_dir": str(memory_dir),
        "output_dir": str(output_dir),
        "importance_floors": IMPORTANCE_FLOORS,
        "entries_seen": len(entries),
        "entries_indexed": len(selected),
        "topic_count": len(gating_report.get("topics", {})),
        "dry_run": args.dry_run,
        "per_topic": gating_report.get("topics", {}),
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    topic_map: dict[str, list[MemoryEntry]] = {}
    for row in selected:
        topic_map.setdefault(row.topic, []).append(row)

    generated_files: list[str] = []
    for topic in sorted(topic_map.keys()):
        filename = _write_topic_file(output_dir, topic, topic_map[topic])
        generated_files.append(filename)

    _write_manifest(output_dir, len(selected), len(topic_map), generated_files, gating_report)

    if args.rebuild_bm25:
        bm25_result = build_bm25_index(memory_dir)
        summary["bm25"] = {k: v for k, v in bm25_result.items() if k != "bm25"}

    log_event(
        "index_rebuild",
        memory_dir=memory_dir,
        results_returned=len(selected),
        extra={"topic_count": len(topic_map)},
    )

    summary["files_written"] = ["SIFTMEM_INDEX.md", *generated_files]
    print(json.dumps(summary))


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
