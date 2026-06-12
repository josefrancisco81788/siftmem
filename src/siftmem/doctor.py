#!/usr/bin/env python3
"""Health checks for a Siftmem memory store."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from siftmem.lib import (
    DEFAULT_JSONL_FILES,
    DEFAULT_MEMORY_DIR,
    load_jsonl_records,
)


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


def _file_mtime(path: Path) -> datetime | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def run_doctor(memory_dir: Path) -> dict:
    issues: list[str] = []
    warnings: list[str] = []
    stats: dict = {"memory_dir": str(memory_dir), "jsonl_files": {}}

    if not memory_dir.exists():
        issues.append(f"memory directory missing: {memory_dir}")

    parse_errors = 0
    entry_count = 0
    for filename in DEFAULT_JSONL_FILES:
        path = memory_dir / filename
        file_info: dict = {"exists": path.exists(), "bytes": 0, "lines": 0}
        if path.exists():
            file_info["bytes"] = path.stat().st_size
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_no, raw in enumerate(handle, start=1):
                    line = raw.strip()
                    if not line:
                        continue
                    file_info["lines"] += 1
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        parse_errors += 1
                        warnings.append(f"{filename}:{line_no} invalid JSON")
                        continue
                    if isinstance(parsed, dict) and parsed.get("topic") and parsed.get("content"):
                        entry_count += 1
        stats["jsonl_files"][filename] = file_info

    stats["entry_count"] = entry_count
    stats["parse_errors"] = parse_errors

    index_dir = memory_dir / "siftmem_index"
    stats["index_dir_exists"] = index_dir.exists()
    if entry_count > 0 and not index_dir.exists():
        warnings.append("entries exist but siftmem_index/ is missing; run siftmem-build-index")

    bm25_path = memory_dir / "siftmem_bm25_index.json"
    stats["bm25_index_exists"] = bm25_path.exists()
    if entry_count > 0 and not bm25_path.exists():
        warnings.append("entries exist but BM25 index is missing; run siftmem-build-index")

    jsonl_mtime = max(
        (m for f in DEFAULT_JSONL_FILES if (m := _file_mtime(memory_dir / f)) is not None),
        default=None,
    )
    index_mtime = _file_mtime(index_dir / "SIFTMEM_INDEX.md") if index_dir.exists() else None
    bm25_mtime = _file_mtime(bm25_path)

    if jsonl_mtime and index_mtime and jsonl_mtime > index_mtime:
        warnings.append("markdown index may be stale (JSONL newer than SIFTMEM_INDEX.md)")
    if jsonl_mtime and bm25_mtime and jsonl_mtime > bm25_mtime:
        warnings.append("BM25 index may be stale (JSONL newer than siftmem_bm25_index.json)")

    provider = os.environ.get("SIFTMEM_LLM_PROVIDER", "none").strip().lower() or "none"
    stats["llm_provider"] = provider
    if provider == "gemini" and not os.environ.get("GEMINI_API_KEY", "").strip():
        warnings.append("SIFTMEM_LLM_PROVIDER=gemini but GEMINI_API_KEY is unset")
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY", "").strip():
        warnings.append("SIFTMEM_LLM_PROVIDER=openai but OPENAI_API_KEY is unset")

    # verify load path works
    try:
        records = load_jsonl_records(memory_dir)
        stats["loaded_records"] = len(records)
    except Exception as exc:  # noqa: BLE001
        issues.append(f"load_jsonl_records failed: {exc}")

    healthy = not issues
    return {
        "ok": healthy,
        "healthy": healthy,
        "issues": issues,
        "warnings": warnings,
        "stats": stats,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Siftmem store health.")
    parser.add_argument("--memory-dir", default=str(DEFAULT_MEMORY_DIR))
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_doctor(Path(args.memory_dir))

    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        status = "healthy" if report["healthy"] else "unhealthy"
        print(f"Siftmem doctor: {status}")
        for issue in report.get("issues", []):
            print(f"  ERROR: {issue}")
        for warning in report.get("warnings", []):
            print(f"  WARN: {warning}")
        stats = report.get("stats", {})
        print(f"  entries: {stats.get('entry_count', 0)}")

    return 0 if report["healthy"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(0)
