#!/usr/bin/env python3
"""High-level Python API for Siftmem."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from siftmem.append import append_entry
from siftmem.lib import (
    DEFAULT_MEMORY_DIR,
    bm25_search,
    build_bm25_index,
    check_dedup,
    load_jsonl_records,
    resolve_superseded_entry_ids,
)


class MemoryStore:
    """Programmatic interface to a Siftmem memory directory."""

    def __init__(self, memory_dir: str | Path | None = None) -> None:
        self.memory_dir = Path(memory_dir or DEFAULT_MEMORY_DIR).expanduser()

    def append(
        self,
        *,
        type: str,
        topic: str,
        content: str,
        importance: float | None = None,
        score_assist: bool = False,
        check_dedup: bool = False,
        supersedes: list[str] | None = None,
        force: bool = False,
        rebuild_index: bool = False,
        dry_run: bool = False,
        extra_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return append_entry(
            entry_type=type,
            topic=topic,
            content=content,
            memory_dir=self.memory_dir,
            importance=importance,
            score_assist=score_assist,
            check_dedup_flag=check_dedup,
            supersedes=supersedes,
            force=force,
            rebuild_index=rebuild_index,
            dry_run=dry_run,
            extra_fields=extra_fields,
        )

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        entry_type: str | None = None,
        topic: str | None = None,
        min_importance: float | None = None,
        explain: bool = False,
    ) -> list[dict[str, Any]]:
        return bm25_search(
            query,
            max_results=max_results,
            memory_dir=self.memory_dir,
            entry_type=entry_type,
            topic=topic,
            min_importance=min_importance,
            explain=explain,
        )

    def rebuild_index(self) -> dict[str, Any]:
        proc = subprocess.run(
            [sys.executable, "-m", "siftmem.build_index", "--memory-dir", str(self.memory_dir)],
            capture_output=True,
            text=True,
        )
        summary: dict[str, Any] = {"ok": proc.returncode == 0, "exit_code": proc.returncode}
        if proc.stdout.strip():
            try:
                import json

                summary["summary"] = json.loads(proc.stdout)
            except json.JSONDecodeError:
                summary["stdout"] = proc.stdout.strip()
        if proc.stderr.strip():
            summary["stderr"] = proc.stderr.strip()
        return summary

    def rebuild_bm25(self) -> dict[str, Any]:
        return build_bm25_index(self.memory_dir)

    def check_dedup(self, entry_type: str, topic: str, content: str) -> dict[str, Any]:
        return check_dedup(entry_type, topic, content, self.memory_dir)

    def stats(self) -> dict[str, Any]:
        records = load_jsonl_records(self.memory_dir)
        superseded = resolve_superseded_entry_ids(records)
        active = [r for r in records if r.entry_id not in superseded]
        by_type: dict[str, int] = {}
        by_topic: dict[str, int] = {}
        for record in active:
            by_type[record.entry_type] = by_type.get(record.entry_type, 0) + 1
            by_topic[record.topic] = by_topic.get(record.topic, 0) + 1
        return {
            "memory_dir": str(self.memory_dir),
            "total_records": len(records),
            "active_records": len(active),
            "superseded_records": len(superseded),
            "by_type": by_type,
            "topic_count": len(by_topic),
        }
