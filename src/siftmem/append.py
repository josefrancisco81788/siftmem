#!/usr/bin/env python3
"""Append entries to canonical Siftmem JSONL files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from siftmem.lib import (
    DEFAULT_MEMORY_DIR,
    TYPE_TO_FILE,
    check_dedup,
    default_importance,
    importance_floor,
    log_event,
    suggest_importance,
    utc_now_z,
)


def _builder_cmd(memory_dir: Path, builder_script: str | None) -> list[str]:
    if builder_script:
        return [sys.executable, builder_script, "--memory-dir", str(memory_dir)]
    return [sys.executable, "-m", "siftmem.build_index", "--memory-dir", str(memory_dir)]


def append_entry(
    *,
    entry_type: str,
    topic: str,
    content: str,
    memory_dir: Path = DEFAULT_MEMORY_DIR,
    importance: float | None = None,
    score_assist: bool = False,
    check_dedup_flag: bool = False,
    supersedes: list[str] | None = None,
    force: bool = False,
    rebuild_index: bool = False,
    builder_script: str | None = None,
    dry_run: bool = False,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one memory entry. Shared by CLI and MemoryStore."""
    if entry_type not in TYPE_TO_FILE:
        raise ValueError(f"unsupported entry type: {entry_type}")

    resolved_importance = importance
    if resolved_importance is None:
        if score_assist:
            resolved_importance = suggest_importance(entry_type, content, topic)
        else:
            resolved_importance = default_importance(entry_type)

    if not (0.0 <= resolved_importance <= 1.0):
        raise ValueError("importance must be between 0.0 and 1.0")

    floor = importance_floor(entry_type)
    below_floor_warning = None
    if resolved_importance < floor and not force:
        below_floor_warning = (
            f"importance {resolved_importance:.2f} is below the floor for type "
            f"'{entry_type}' ({floor:.2f}). Entry may not be indexed."
        )

    dedup: dict[str, Any] = {"action": "append"}
    if check_dedup_flag:
        dedup = check_dedup(entry_type, topic, content, memory_dir)
        if dedup["action"] == "skip":
            log_event(
                "dedup_hit",
                memory_dir=memory_dir,
                topic=topic,
                extra={"similarity": dedup.get("similarity"), "matched": dedup.get("matched")},
                dry_run=dry_run,
            )
            return {"ok": True, "skipped": True, "reason": "dedup", "dedup": dedup}

    entry: dict[str, Any] = {
        "timestamp": utc_now_z(),
        "type": entry_type,
        "topic": topic,
        "content": content,
        "importance": float(resolved_importance),
    }
    if extra_fields:
        entry.update(extra_fields)

    supersedes_refs = [str(ref).strip() for ref in (supersedes or []) if str(ref).strip()]
    if dedup.get("action") == "supersede" and dedup.get("supersedes"):
        supersedes_refs.append(str(dedup.get("supersedes")).strip())
    if supersedes_refs:
        unique_refs: list[str] = []
        seen_refs: set[str] = set()
        for ref in supersedes_refs:
            if ref and ref not in seen_refs:
                seen_refs.add(ref)
                unique_refs.append(ref)
        entry["supersedes"] = unique_refs[0] if len(unique_refs) == 1 else unique_refs
    if dedup.get("conflict"):
        entry["conflict"] = True

    target = memory_dir / TYPE_TO_FILE[entry_type]
    output: dict[str, Any] = {
        "ok": True,
        "appended_to": str(target),
        "entry": entry,
        "index_rebuilt": False,
        "dedup": dedup,
        "below_floor_warning": below_floor_warning,
    }

    if dry_run:
        output["dry_run"] = True
        return output

    memory_dir.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    log_event("append", memory_dir=memory_dir, topic=topic, extra={"type": entry_type})

    if dedup.get("conflict"):
        output["conflict_warning"] = (
            f"Conflicting decision detected on topic '{topic}'. Review recommended."
        )

    if rebuild_index:
        proc = subprocess.run(
            _builder_cmd(memory_dir, builder_script),
            text=True,
            capture_output=True,
        )
        output["index_rebuilt"] = proc.returncode == 0
        output["builder_exit_code"] = proc.returncode
        output["builder_stdout"] = proc.stdout.strip()
        output["builder_stderr"] = proc.stderr.strip()
        if proc.returncode != 0:
            output["ok"] = False

    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a single entry to Siftmem JSONL.")
    parser.add_argument("--type", required=True, choices=sorted(TYPE_TO_FILE.keys()))
    parser.add_argument("--topic", required=True)
    parser.add_argument("--content", required=True)
    parser.add_argument("--importance", type=float, default=None)
    parser.add_argument(
        "--score-assist",
        action="store_true",
        help="Refine importance with heuristics and optional LLM when omitted.",
    )
    parser.add_argument("--memory-dir", default=str(DEFAULT_MEMORY_DIR))
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument(
        "--builder-script",
        default=None,
        help="Optional path to build_index script; default uses python -m siftmem.build_index",
    )
    parser.add_argument(
        "--supersedes",
        action="append",
        default=[],
        help="Timestamp or entry_id of prior records this entry supersedes. Repeatable.",
    )
    parser.add_argument(
        "--check-dedup",
        action="store_true",
        help="Skip near-duplicate writes (default off for manual append).",
    )
    parser.add_argument("--force", action="store_true", help="Suppress below-floor warnings.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    memory_dir = Path(args.memory_dir)

    try:
        output = append_entry(
            entry_type=args.type,
            topic=args.topic,
            content=args.content,
            memory_dir=memory_dir,
            importance=args.importance,
            score_assist=args.score_assist,
            check_dedup_flag=args.check_dedup,
            supersedes=args.supersedes,
            force=args.force,
            rebuild_index=args.rebuild_index,
            builder_script=args.builder_script,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if output.get("below_floor_warning") and not args.force:
        print(f"WARNING: {output['below_floor_warning']} Pass --force to suppress.", file=sys.stderr)
    if output.get("conflict_warning"):
        print(f"[Siftmem] {output['conflict_warning']}", file=sys.stderr)

    print(json.dumps(output, ensure_ascii=False))
    if output.get("builder_exit_code", 0) not in (0, None):
        raise SystemExit(output["builder_exit_code"])


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
