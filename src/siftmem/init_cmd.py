#!/usr/bin/env python3
"""Bootstrap a new Siftmem memory store."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from siftmem.lib import DEFAULT_JSONL_FILES, DEFAULT_MEMORY_DIR, utc_now_z

SAMPLE_ENTRY = {
    "timestamp": None,  # filled at runtime
    "type": "decision",
    "topic": "getting-started",
    "content": (
        "Use siftmem-append for durable writes and siftmem-search for keyword recall. "
        "Rebuild the index after batch writes with siftmem-build-index."
    ),
    "importance": 0.9,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize a Siftmem memory store.")
    parser.add_argument("--memory-dir", default=str(DEFAULT_MEMORY_DIR))
    parser.add_argument("--force", action="store_true", help="Overwrite sample entry if store exists.")
    parser.add_argument("--skip-index", action="store_true", help="Do not run build_index after init.")
    return parser.parse_args()


def init_store(memory_dir: Path, *, force: bool = False) -> dict:
    memory_dir.mkdir(parents=True, exist_ok=True)
    created_files: list[str] = []
    sample_written = False

    for filename in DEFAULT_JSONL_FILES:
        path = memory_dir / filename
        if not path.exists():
            path.touch()
            created_files.append(filename)

    decisions_path = memory_dir / "decisions.jsonl"
    has_content = decisions_path.exists() and decisions_path.stat().st_size > 0
    if not has_content or force:
        entry = {**SAMPLE_ENTRY, "timestamp": utc_now_z()}
        with decisions_path.open("a" if has_content and force else "w", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        sample_written = True

    return {
        "ok": True,
        "memory_dir": str(memory_dir),
        "created_files": created_files,
        "sample_written": sample_written,
    }


def main() -> int:
    args = _parse_args()
    memory_dir = Path(args.memory_dir)
    result = init_store(memory_dir, force=args.force)
    print(json.dumps(result, indent=2))

    if not args.skip_index and result.get("sample_written"):
        proc = subprocess.run(
            [sys.executable, "-m", "siftmem.build_index", "--memory-dir", str(memory_dir)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            return proc.returncode

    print(
        "\nNext steps:\n"
        f"  export SIFTMEM_MEMORY_DIR={memory_dir}\n"
        "  siftmem-append --type fact --topic my-topic --content \"Your memory here.\"\n"
        "  siftmem-search \"my-topic\" --json\n"
        "  siftmem-doctor",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(0)
