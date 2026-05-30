#!/usr/bin/env python3
"""Extract structured memories from recent agent sessions (Phase 2 H4)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from siftmem_lib import (
    DEFAULT_MEMORY_DIR,
    DEFAULT_REPO,
    EXTRACTION_PROMPT,
    check_dedup,
    gemini_generate_json,
    log_event,
    utc_now_z,
)

DEFAULT_SESSIONS_DIR = DEFAULT_REPO / "agents" / "main" / "sessions"
APPEND_SCRIPT = Path(__file__).resolve().parent / "siftmem_append.py"
BUILDER_SCRIPT = Path(__file__).resolve().parent / "siftmem_build_index.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture memories from recent sessions.")
    parser.add_argument("--sessions-dir", default=str(DEFAULT_SESSIONS_DIR))
    parser.add_argument("--memory-dir", default=str(DEFAULT_MEMORY_DIR))
    parser.add_argument("--max-sessions", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true", default=True)
    parser.add_argument("--no-rebuild-index", action="store_false", dest="rebuild_index")
    return parser.parse_args()


def _extract_transcript(path: Path, max_turns: int) -> str:
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "message":
                continue
            msg = event.get("message") or {}
            role = msg.get("role", "unknown")
            parts = msg.get("content") or []
            texts = []
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = str(part.get("text", "")).strip()
                    if text:
                        texts.append(text)
            if texts:
                lines.append(f"{role}: " + " ".join(texts))
    if len(lines) > max_turns:
        lines = lines[-max_turns:]
    return "\n".join(lines)


def _recent_session_files(sessions_dir: Path, limit: int) -> list[Path]:
    candidates = [
        p
        for p in sessions_dir.glob("*.jsonl")
        if p.is_file() and not p.name.endswith(".trajectory.jsonl")
    ]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[:limit]


def _append_entry(
    entry: dict,
    memory_dir: Path,
    *,
    dry_run: bool,
) -> dict:
    entry_type = str(entry.get("type", "fact"))
    topic = str(entry.get("topic", "")).strip()
    content = str(entry.get("content", "")).strip()
    try:
        importance = float(entry.get("importance", 0.7))
    except (TypeError, ValueError):
        importance = 0.7

    if not topic or not content:
        return {"ok": False, "error": "missing topic or content"}

    dedup = check_dedup(entry_type, topic, content, memory_dir)
    if dedup["action"] == "skip":
        log_event("dedup_hit", memory_dir=memory_dir, topic=topic, extra=dedup, dry_run=dry_run)
        return {"ok": True, "skipped": True, "dedup": dedup, "entry": entry}

    if dry_run:
        return {"ok": True, "dry_run": True, "would_append": entry, "dedup": dedup}

    cmd = [
        "python3",
        str(APPEND_SCRIPT),
        "--type",
        entry_type,
        "--topic",
        topic,
        "--content",
        content,
        "--importance",
        str(importance),
        "--memory-dir",
        str(memory_dir),
    ]
    if dedup.get("supersedes"):
        pass  # append handles via JSONL fields — add supersedes in direct write below

    # Direct append to preserve supersedes/conflict fields
    from siftmem_lib import TYPE_TO_FILE

    target = memory_dir / TYPE_TO_FILE[entry_type]
    payload = {
        "timestamp": utc_now_z(),
        "type": entry_type,
        "topic": topic,
        "content": content,
        "importance": importance,
        "source": "session_capture",
    }
    if dedup.get("supersedes"):
        payload["supersedes"] = dedup["supersedes"]
    if dedup.get("conflict"):
        payload["conflict"] = True

    memory_dir.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    log_event("append", memory_dir=memory_dir, topic=topic, extra={"source": "capture"})
    return {"ok": True, "appended": True, "entry": payload, "dedup": dedup}


def main() -> int:
    args = _parse_args()
    sessions_dir = Path(args.sessions_dir)
    memory_dir = Path(args.memory_dir)

    if not sessions_dir.exists():
        print(json.dumps({"ok": False, "error": f"sessions dir missing: {sessions_dir}"}))
        return 1

    session_files = _recent_session_files(sessions_dir, args.max_sessions)
    transcripts: list[str] = []
    for path in session_files:
        text = _extract_transcript(path, args.max_turns)
        if text.strip():
            transcripts.append(f"## Session {path.name}\n{text}")

    if not transcripts:
        print(json.dumps({"ok": True, "extracted": 0, "reason": "no transcript text"}))
        return 0

    combined = "\n\n".join(transcripts)
    extracted = gemini_generate_json(EXTRACTION_PROMPT, combined)
    if extracted is None:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "gemini extraction unavailable (no API key or API failure)",
                    "sessions_scanned": len(session_files),
                }
            )
        )
        return 0

    if isinstance(extracted, dict):
        entries = [extracted]
    elif isinstance(extracted, list):
        entries = extracted
    else:
        print(json.dumps({"ok": False, "error": "unexpected extraction shape"}))
        return 1

    results = []
    appended = 0
    for item in entries:
        if not isinstance(item, dict):
            continue
        result = _append_entry(item, memory_dir, dry_run=args.dry_run)
        results.append(result)
        if result.get("appended"):
            appended += 1

    if args.rebuild_index and not args.dry_run and appended > 0:
        subprocess.run(
            ["python3", str(BUILDER_SCRIPT), "--memory-dir", str(memory_dir)],
            check=False,
        )

    summary = {
        "ok": True,
        "sessions_scanned": len(session_files),
        "extracted_candidates": len(entries),
        "appended": appended,
        "dry_run": args.dry_run,
        "results": results,
    }
    log_event(
        "capture_run",
        memory_dir=memory_dir,
        results_returned=appended,
        extra={"candidates": len(entries)},
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(0)
