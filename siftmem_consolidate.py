#!/usr/bin/env python3
"""Weekly synthesis of dense Siftmem topics (Phase 2 H10)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from siftmem_lib import (
    DEFAULT_JSONL_FILES,
    DEFAULT_MEMORY_DIR,
    TYPE_TO_FILE,
    gemini_generate_json,
    log_event,
    utc_now_z,
)

BUILDER_SCRIPT = Path(__file__).resolve().parent / "siftmem_build_index.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly Siftmem topic consolidation.")
    parser.add_argument("--memory-dir", default=str(DEFAULT_MEMORY_DIR))
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--min-new-entries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true", default=True)
    return parser.parse_args()


def _parse_ts(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_recent_by_topic(memory_dir: Path, days: int) -> dict[str, list[dict]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    by_topic: dict[str, list[dict]] = {}
    for filename in DEFAULT_JSONL_FILES:
        path = memory_dir / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or row.get("superseded"):
                    continue
                ts = _parse_ts(str(row.get("timestamp", "")))
                if ts is None or ts < cutoff:
                    continue
                topic = str(row.get("topic", "")).strip()
                if not topic:
                    continue
                row["_source"] = f"{filename}:{line_no}"
                by_topic.setdefault(topic, []).append(row)
    return by_topic


def _annotate_superseded(memory_dir: Path, sources: list[dict], *, dry_run: bool) -> int:
    if dry_run:
        return len(sources)
    updated = 0
    by_file: dict[str, list[tuple[int, dict]]] = {}
    for row in sources:
        source = str(row.get("_source", ""))
        if ":" not in source:
            continue
        filename, line_s = source.split(":", 1)
        by_file.setdefault(filename, []).append((int(line_s), row))

    for filename, items in by_file.items():
        path = memory_dir / filename
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_no, _row in items:
            idx = line_no - 1
            if idx < 0 or idx >= len(lines) or not lines[idx].strip():
                continue
            try:
                parsed = json.loads(lines[idx])
            except json.JSONDecodeError:
                continue
            if parsed.get("superseded"):
                continue
            parsed["superseded"] = True
            lines[idx] = json.dumps(parsed, ensure_ascii=False)
            updated += 1
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return updated


def main() -> int:
    args = _parse_args()
    memory_dir = Path(args.memory_dir)
    by_topic = _load_recent_by_topic(memory_dir, args.days)

    synthesized = 0
    report: list[dict] = []

    for topic, rows in sorted(by_topic.items()):
        if len(rows) < args.min_new_entries:
            continue
        prompt = (
            "Synthesize these recent memory entries into one durable summary entry. "
            'Respond with JSON: {"type":"lesson|decision|fact","content":"...","importance":0.9}'
        )
        payload = gemini_generate_json(prompt, json.dumps(rows, ensure_ascii=False))
        if not isinstance(payload, dict):
            report.append({"topic": topic, "ok": False, "error": "synthesis failed"})
            continue

        entry_type = str(payload.get("type", "lesson"))
        content = str(payload.get("content", "")).strip()
        try:
            importance = float(payload.get("importance", 0.9))
        except (TypeError, ValueError):
            importance = 0.9
        if not content:
            continue

        synthesis = {
            "timestamp": utc_now_z(),
            "type": entry_type,
            "topic": topic,
            "content": content,
            "importance": min(1.0, max(0.0, importance)),
            "synthesised_from": [str(r.get("timestamp", "")) for r in rows],
        }

        if not args.dry_run:
            target = memory_dir / TYPE_TO_FILE.get(entry_type, "lessons.jsonl")
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(synthesis, ensure_ascii=False) + "\n")
            _annotate_superseded(memory_dir, rows, dry_run=False)

        synthesized += 1
        report.append({"topic": topic, "ok": True, "sources": len(rows), "dry_run": args.dry_run})

    if synthesized and args.rebuild_index and not args.dry_run:
        import subprocess

        subprocess.run(
            ["python3", str(BUILDER_SCRIPT), "--memory-dir", str(memory_dir)],
            check=False,
        )

    log_event(
        "consolidate_run",
        memory_dir=memory_dir,
        results_returned=synthesized,
        dry_run=args.dry_run,
    )
    print(json.dumps({"ok": True, "synthesized": synthesized, "report": report}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(0)
