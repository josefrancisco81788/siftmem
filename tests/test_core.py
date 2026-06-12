from __future__ import annotations

import json
from pathlib import Path

import pytest

from siftmem.append import append_entry
from siftmem.doctor import run_doctor
from siftmem.init_cmd import init_store
from siftmem.lib import (
    IMPORTANCE_FLOORS,
    bm25_search,
    default_importance,
    heuristic_importance,
    importance_floor,
    load_jsonl_records,
    resolve_superseded_entry_ids,
)
from siftmem.store import MemoryStore


def test_default_importance_uses_floor_plus_margin() -> None:
    assert default_importance("decision") == pytest.approx(0.90)
    assert default_importance("fact") == pytest.approx(0.65)


def test_heuristic_importance_boosts_imperatives() -> None:
    base = heuristic_importance("fact", "A note about tooling.", "tooling")
    boosted = heuristic_importance("fact", "Always validate inputs before indexing.", "tooling")
    assert boosted >= base


def test_append_without_importance(memory_dir: Path) -> None:
    result = append_entry(
        entry_type="fact",
        topic="test",
        content="hello world",
        memory_dir=memory_dir,
        rebuild_index=True,
    )
    assert result["ok"] is True
    assert result["entry"]["importance"] == default_importance("fact")

    facts = (memory_dir / "facts.jsonl").read_text(encoding="utf-8")
    assert "hello world" in facts


def test_dedup_skip(memory_dir: Path) -> None:
    append_entry(
        entry_type="fact",
        topic="dup-topic",
        content="same content here",
        memory_dir=memory_dir,
        importance=0.7,
    )
    second = append_entry(
        entry_type="fact",
        topic="dup-topic",
        content="same content here",
        memory_dir=memory_dir,
        importance=0.7,
        check_dedup_flag=True,
    )
    assert second.get("skipped") is True


def test_supersession_resolution(memory_dir: Path) -> None:
    first = append_entry(
        entry_type="decision",
        topic="policy",
        content="Use approach A for exports.",
        memory_dir=memory_dir,
        importance=0.9,
    )
    ts = first["entry"]["timestamp"]
    append_entry(
        entry_type="decision",
        topic="policy",
        content="Use approach B for exports instead.",
        memory_dir=memory_dir,
        importance=0.92,
        supersedes=[ts],
    )
    records = load_jsonl_records(memory_dir)
    superseded = resolve_superseded_entry_ids(records)
    assert len(superseded) >= 1


def test_index_below_floor_fallback(memory_dir: Path) -> None:
    append_entry(
        entry_type="fact",
        topic="lonely-topic",
        content="low importance lone fact",
        memory_dir=memory_dir,
        importance=0.1,
        force=True,
        rebuild_index=True,
    )
    index_file = memory_dir / "siftmem_index" / "topic__lonely-topic.md"
    assert index_file.exists()
    text = index_file.read_text(encoding="utf-8")
    assert "below-threshold" in text or "low importance lone fact" in text


def test_bm25_search_round_trip(memory_dir: Path) -> None:
    store = MemoryStore(memory_dir)
    store.append(
        type="decision",
        topic="onboarding",
        content="Always rebuild the index after batch writes.",
        importance=0.9,
        rebuild_index=True,
    )
    hits = store.search("onboarding rebuild", max_results=3)
    assert hits
    assert hits[0]["topic"] == "onboarding"


def test_search_type_filter(memory_dir: Path) -> None:
    store = MemoryStore(memory_dir)
    store.append(type="fact", topic="mixed", content="fact about paths /tmp/foo", importance=0.7, rebuild_index=True)
    store.append(
        type="decision",
        topic="mixed",
        content="decision about paths /tmp/foo",
        importance=0.9,
        rebuild_index=True,
    )
    hits = store.search("paths tmp foo", entry_type="decision", max_results=5)
    assert hits
    assert all(row["type"] == "decision" for row in hits)


def test_importance_weighted_ranking(memory_dir: Path) -> None:
    store = MemoryStore(memory_dir)
    store.append(
        type="fact",
        topic="ranking",
        content="shared keyword alpha",
        importance=0.6,
        rebuild_index=True,
    )
    store.append(
        type="fact",
        topic="ranking",
        content="shared keyword alpha",
        importance=0.95,
        rebuild_index=True,
    )
    hits = bm25_search("shared keyword alpha", memory_dir=memory_dir, explain=True, max_results=5)
    assert len(hits) >= 2
    assert hits[0]["importance"] >= hits[1]["importance"]


def test_memory_store_stats(memory_dir: Path) -> None:
    store = MemoryStore(memory_dir)
    store.append(type="lesson", topic="stats", content="count me", importance=0.75)
    stats = store.stats()
    assert stats["active_records"] == 1
    assert stats["by_type"]["lesson"] == 1


def test_init_store_writes_sample(memory_dir: Path) -> None:
    result = init_store(memory_dir)
    assert result["sample_written"] is True
    decisions = (memory_dir / "decisions.jsonl").read_text(encoding="utf-8")
    assert "getting-started" in decisions


def test_doctor_healthy_after_init(memory_dir: Path) -> None:
    init_store(memory_dir)
    report = run_doctor(memory_dir)
    assert report["healthy"] is True


def test_importance_floors_exported() -> None:
    assert IMPORTANCE_FLOORS["decision"] == importance_floor("decision")
