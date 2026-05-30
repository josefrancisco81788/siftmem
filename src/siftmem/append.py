#!/usr/bin/env python3
"""Append entries to canonical Siftmem JSONL files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from siftmem.lib import (
    DEFAULT_MEMORY_DIR,
    TYPE_TO_FILE,
    check_dedup,
    importance_floor,
    log_event,
    suggest_importance,
    utc_now_z,
)

def _builder_cmd(memory_dir: Path, builder_script: str | None) -> list[str]:
    if builder_script:
        return [sys.executable, builder_script, "--memory-dir", str(memory_dir)]
    return [sys.executable, "-m", "siftmem.build_index", "--memory-dir", str(memory_dir)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a single entry to Siftmem JSONL.")
    parser.add_argument("--type", required=True, choices=sorted(TYPE_TO_FILE.keys()))
    parser.add_argument("--topic", required=True)
    parser.add_argument("--content", required=True)
    parser.add_argument("--importance", type=float, default=None)
    parser.add_argument("--score-assist", action="store_true", help="Gemini suggests importance if omitted.")
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

    importance = args.importance
    if importance is None:
        if not args.score_assist:
            raise SystemExit("Provide --importance or use --score-assist.")
        importance = suggest_importance(args.type, args.content, args.topic)

    if not (0.0 <= importance <= 1.0):
        raise SystemExit("--importance must be between 0.0 and 1.0")

    floor = importance_floor(args.type)
    if importance < floor and not args.force:
        print(
            f"WARNING: importance {importance:.2f} is below the floor for type "
            f"'{args.type}' ({floor:.2f}). Entry may not be indexed. Pass --force to suppress.",
            file=sys.stderr,
        )

    dedup = {"action": "append"}
    if args.check_dedup:
        dedup = check_dedup(args.type, args.topic, args.content, memory_dir)
        if dedup["action"] == "skip":
            log_event(
                "dedup_hit",
                memory_dir=memory_dir,
                topic=args.topic,
                extra={"similarity": dedup.get("similarity"), "matched": dedup.get("matched")},
                dry_run=args.dry_run,
            )
            print(json.dumps({"ok": True, "skipped": True, "reason": "dedup", "dedup": dedup}))
            return

    entry: dict = {
        "timestamp": utc_now_z(),
        "type": args.type,
        "topic": args.topic,
        "content": args.content,
        "importance": float(importance),
    }
    supersedes_refs = [str(ref).strip() for ref in (args.supersedes or []) if str(ref).strip()]
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

    target = memory_dir / TYPE_TO_FILE[args.type]
    output = {"appended_to": str(target), "entry": entry, "index_rebuilt": False, "dedup": dedup}

    if args.dry_run:
        output["dry_run"] = True
        print(json.dumps(output, ensure_ascii=False))
        return

    memory_dir.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    log_event("append", memory_dir=memory_dir, topic=args.topic, extra={"type": args.type})

    if dedup.get("conflict"):
        print(
            "[Siftmem] Conflicting decision detected on topic "
            f"'{args.topic}'. Review recommended.",
            file=sys.stderr,
        )

    if args.rebuild_index:
        proc = subprocess.run(
            _builder_cmd(memory_dir, args.builder_script),
            text=True,
            capture_output=True,
        )
        output["index_rebuilt"] = proc.returncode == 0
        output["builder_exit_code"] = proc.returncode
        output["builder_stdout"] = proc.stdout.strip()
        output["builder_stderr"] = proc.stderr.strip()
        if proc.returncode != 0:
            print(json.dumps(output))
            raise SystemExit(proc.returncode)

    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
