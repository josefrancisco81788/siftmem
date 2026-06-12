#!/usr/bin/env python3
"""Shared Siftmem utilities."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

DEFAULT_MEMORY_DIR = Path(os.environ.get("SIFTMEM_MEMORY_DIR") or str(Path.home() / ".siftmem" / "memory"))
DEFAULT_REPO = Path(os.environ.get("SIFTMEM_HOME") or str(Path.home() / ".siftmem"))
RETRIEVAL_LOG = DEFAULT_MEMORY_DIR / "retrieval-log.jsonl"
BM25_INDEX_PATH = DEFAULT_MEMORY_DIR / "siftmem_bm25_index.json"
EMBEDDING_CACHE_DIR = DEFAULT_MEMORY_DIR / ".embedding-cache"

IMPORTANCE_FLOORS: dict[str, float] = {
    "decision": 0.85,
    "preference": 0.80,
    "lesson": 0.70,
    "fact": 0.60,
}
TOPIC_MAX_PER_TYPE: dict[str, int] = {
    "fact": 5,
    "decision": 10,
    "preference": 10,
    "lesson": 10,
}
DEFAULT_TOPIC_MAX = 10

TYPE_TO_FILE = {
    "fact": "facts.jsonl",
    "decision": "decisions.jsonl",
    "preference": "preferences.jsonl",
    "lesson": "lessons.jsonl",
}

DEFAULT_JSONL_FILES = (
    "facts.jsonl",
    "decisions.jsonl",
    "preferences.jsonl",
    "lessons.jsonl",
    "bootstrap.jsonl",
)

EXTRACTION_PROMPT = """You are a memory extraction agent. Given a conversation transcript, extract structured
memories. For each memory, output JSON with keys: type (fact|decision|preference|lesson),
topic (kebab-case, 2-4 words), content (1-3 sentences, specific and actionable),
importance (float 0.0-1.0, see rubric below).

Importance rubric:
- decision: 0.85-1.0 (irreversible, high-stakes choices)
- preference: 0.80-0.95 (stable user or system preferences)
- lesson: 0.70-0.85 (operational lessons, failure post-mortems)
- fact: 0.50-0.80 (transient facts; promote to 0.8+ only if cross-session relevant)

Output a JSON array only. No preamble. No markdown fences."""


@dataclass
class MemoryRecord:
    timestamp: str
    entry_type: str
    topic: str
    content: str
    importance: float
    source_file: str
    line_no: int
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def entry_id(self) -> str:
        return f"{self.source_file}:{self.line_no}"


def utc_now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def importance_floor(entry_type: str) -> float:
    return IMPORTANCE_FLOORS.get(entry_type, 0.8)


def default_importance(entry_type: str) -> float:
    return min(1.0, importance_floor(entry_type) + 0.05)


_IMPERATIVE_RE = re.compile(
    r"\b(always|never|must|required|do not|don't|shall|avoid)\b",
    re.I,
)
_TRANSIENT_RE = re.compile(
    r"\b(today|yesterday|currently|right now|this week|temporary|for now)\b",
    re.I,
)


def heuristic_importance(entry_type: str, content: str, topic: str) -> float:
    """Rule-based importance estimate; no LLM required."""
    score = default_importance(entry_type)
    text = f"{topic} {content}".lower()

    if _IMPERATIVE_RE.search(text):
        score = min(1.0, score + 0.05)
    if entry_type == "decision" and re.search(r"\b(instead|rather than|chose|decided)\b", text):
        score = min(1.0, score + 0.03)
    if _TRANSIENT_RE.search(text):
        score = max(importance_floor(entry_type), score - 0.08)
    if len(content.strip()) < 30:
        score = max(importance_floor(entry_type), score - 0.05)
    if len(content.strip()) > 400:
        score = min(1.0, score + 0.02)

    return round(min(1.0, max(0.0, score)), 2)


def log_event(
    event: str,
    *,
    memory_dir: Path = DEFAULT_MEMORY_DIR,
    query: str | None = None,
    results_returned: int = 0,
    top_score: float = 0.0,
    tokens_estimated: int = 0,
    topic: str | None = None,
    extra: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> None:
    payload: dict[str, Any] = {
        "timestamp": utc_now_z(),
        "event": event,
        "query": query,
        "results_returned": results_returned,
        "top_score": top_score,
        "tokens_estimated": tokens_estimated,
        "topic": topic,
    }
    if extra:
        payload.update(extra)
    if dry_run:
        return
    path = memory_dir / "retrieval-log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def normalize_text(text: str, max_chars: int | None = None) -> str:
    cleaned = " ".join(str(text).split())
    if max_chars is None or len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"


def text_similarity(a: str, b: str) -> float:
    left = normalize_text(a).lower()
    right = normalize_text(b).lower()
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).lower().encode("utf-8")).hexdigest()


def _supersedes_refs(raw: dict[str, Any]) -> list[str]:
    value = raw.get("supersedes")
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        refs = []
        for item in value:
            text = str(item or "").strip()
            if text:
                refs.append(text)
        return refs
    return []


def resolve_superseded_entry_ids(records: list[MemoryRecord]) -> set[str]:
    by_entry_id = {record.entry_id: record for record in records}
    by_timestamp: dict[str, list[MemoryRecord]] = {}
    superseded: set[str] = set()

    for record in records:
        if record.raw.get("superseded") is True:
            superseded.add(record.entry_id)
        timestamp = str(record.timestamp or "").strip()
        if timestamp:
            by_timestamp.setdefault(timestamp, []).append(record)

    for record in records:
        for ref in _supersedes_refs(record.raw):
            if ref in by_entry_id:
                superseded.add(ref)
            for matched in by_timestamp.get(ref, []):
                superseded.add(matched.entry_id)
    return superseded


def load_jsonl_records(
    memory_dir: Path,
    filenames: Iterable[str] | None = None,
    *,
    resolve_supersession: bool = False,
    include_superseded: bool = True,
) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    for filename in filenames or DEFAULT_JSONL_FILES:
        path = memory_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(parsed, dict):
                    continue
                topic = str(parsed.get("topic", "")).strip()
                content = str(parsed.get("content", "")).strip()
                if not topic or not content:
                    continue
                records.append(
                    MemoryRecord(
                        timestamp=str(parsed.get("timestamp", "")),
                        entry_type=str(parsed.get("type", "fact")),
                        topic=topic,
                        content=content,
                        importance=float(parsed.get("importance", 0.5) or 0.5),
                        source_file=filename,
                        line_no=line_no,
                        raw=parsed,
                    )
                )
    if resolve_supersession or not include_superseded:
        superseded_ids = resolve_superseded_entry_ids(records)
        if not include_superseded:
            records = [record for record in records if record.entry_id not in superseded_ids]
    return records


def load_topic_records(
    memory_dir: Path,
    entry_type: str,
    topic: str,
    *,
    resolve_supersession: bool = False,
    include_superseded: bool = True,
) -> list[MemoryRecord]:
    filename = TYPE_TO_FILE.get(entry_type)
    if not filename:
        return []
    return [
        r
        for r in load_jsonl_records(
            memory_dir,
            [filename],
            resolve_supersession=resolve_supersession,
            include_superseded=include_superseded,
        )
        if r.topic == topic
    ]


def gemini_generate_json(prompt: str, user_content: str, *, model: str = "gemini-2.5-flash") -> Any | None:
    """Deprecated: use siftmem.llm.generate_json with provider='gemini'."""
    from siftmem.llm import gemini_generate_json as _gemini_generate_json

    return _gemini_generate_json(prompt, user_content, model=model)


def suggest_importance(
    entry_type: str,
    content: str,
    topic: str,
    *,
    use_llm: bool = True,
) -> float:
    """Score importance via heuristic; optionally refine with LLM when available."""
    score = heuristic_importance(entry_type, content, topic)
    if not use_llm:
        return score

    from siftmem.llm import generate_json, resolve_provider

    if resolve_provider() == "none":
        return score

    result = generate_json(
        "Suggest a single importance score between 0 and 1 for this memory entry. "
        "Respond with JSON: {\"importance\": 0.75}",
        json.dumps({"type": entry_type, "topic": topic, "content": content}),
    )
    if isinstance(result, dict) and "importance" in result:
        try:
            return float(max(0.0, min(1.0, float(result["importance"]))))
        except (TypeError, ValueError):
            pass
    return score


def check_dedup(
    entry_type: str,
    topic: str,
    content: str,
    memory_dir: Path = DEFAULT_MEMORY_DIR,
    *,
    dedup_threshold: float = 0.92,
    soft_threshold: float = 0.80,
) -> dict[str, Any]:
    """Return action: append | skip | supersede with optional supersedes timestamp."""
    existing = load_topic_records(memory_dir, entry_type, topic)
    best_sim = 0.0
    best: MemoryRecord | None = None
    for row in existing:
        sim = text_similarity(content, row.content)
        if sim > best_sim:
            best_sim = sim
            best = row
    if best_sim >= dedup_threshold:
        return {"action": "skip", "similarity": best_sim, "matched": best.entry_id if best else None}
    if best_sim >= soft_threshold and best:
        return {
            "action": "supersede",
            "similarity": best_sim,
            "supersedes": best.timestamp,
            "matched": best.entry_id,
        }
    if entry_type == "decision" and existing:
        contradictions = [
            r for r in existing if text_similarity(content, r.content) < 0.5
        ]
        if contradictions:
            return {
                "action": "append_conflict",
                "similarity": best_sim,
                "conflict": True,
            }
    return {"action": "append", "similarity": best_sim}


def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_/\\.:-]+", text.lower())


def _simple_token_score(query_tokens: list[str], doc_tokens: list[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    doc_set = set(doc_tokens)
    hits = sum(1 for token in query_tokens if token in doc_set)
    return hits / max(len(query_tokens), 1)


def bm25_index_path(memory_dir: Path = DEFAULT_MEMORY_DIR) -> Path:
    return memory_dir / "siftmem_bm25_index.json"


def build_bm25_index(memory_dir: Path = DEFAULT_MEMORY_DIR) -> dict[str, Any]:
    index_path = bm25_index_path(memory_dir)
    records = load_jsonl_records(memory_dir, resolve_supersession=True, include_superseded=False)
    if not records:
        payload = {
            "generated_at_utc": utc_now_z(),
            "corpus_size": 0,
            "documents": [],
            "corpus_tokens": [],
            "backend": "none",
        }
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return {"ok": True, "corpus_size": 0, "path": str(index_path)}

    corpus_tokens = [tokenize_for_bm25(f"{r.topic} {r.content}") for r in records]
    documents = [
        {
            "id": r.entry_id,
            "topic": r.topic,
            "type": r.entry_type,
            "content": r.content,
            "importance": r.importance,
            "timestamp": r.timestamp,
        }
        for r in records
    ]
    backend = "simple"
    try:
        from rank_bm25 import BM25Okapi  # noqa: F401

        backend = "rank_bm25"
    except ImportError:
        pass

    payload = {
        "generated_at_utc": utc_now_z(),
        "corpus_size": len(documents),
        "documents": documents,
        "corpus_tokens": corpus_tokens,
        "backend": backend,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "corpus_size": len(documents), "path": str(index_path), "backend": backend}


def _importance_boost(importance: float) -> float:
    return 0.5 + 0.5 * min(1.0, max(0.0, importance))


def _apply_importance_to_score(raw_score: float, importance: float) -> float:
    """Blend BM25 score with importance; handles negative BM25 on small corpora."""
    clamped = min(1.0, max(0.0, importance))
    boost = _importance_boost(clamped)
    if raw_score > 0:
        return float(raw_score) * boost
    return float(raw_score) - (1.0 - clamped) * 0.05


def _apply_search_filters(
    doc: dict[str, Any],
    *,
    entry_type: str | None,
    topic: str | None,
    min_importance: float | None,
) -> bool:
    if entry_type and str(doc.get("type", "")) != entry_type:
        return False
    if topic and str(doc.get("topic", "")) != topic:
        return False
    if min_importance is not None:
        try:
            if float(doc.get("importance", 0.0)) < min_importance:
                return False
        except (TypeError, ValueError):
            return False
    return True


def bm25_search(
    query: str,
    *,
    max_results: int = 5,
    memory_dir: Path = DEFAULT_MEMORY_DIR,
    entry_type: str | None = None,
    topic: str | None = None,
    min_importance: float | None = None,
    explain: bool = False,
) -> list[dict[str, Any]]:
    index_path = bm25_index_path(memory_dir)
    if not index_path.exists():
        build_bm25_index(memory_dir)
    if not index_path.exists():
        return []

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    documents = payload.get("documents") or []
    corpus_tokens = payload.get("corpus_tokens") or []
    if not documents or not corpus_tokens:
        return []

    query_tokens = tokenize_for_bm25(query)
    raw_scores: list[float] = []
    try:
        from rank_bm25 import BM25Okapi

        bm25 = BM25Okapi(corpus_tokens)
        raw_scores = list(bm25.get_scores(query_tokens))
    except ImportError:
        raw_scores = [_simple_token_score(query_tokens, doc_tokens) for doc_tokens in corpus_tokens]

    scored: list[tuple[dict[str, Any], float, float]] = []
    for doc, raw_score in zip(documents, raw_scores):
        if raw_score == 0:
            continue
        if not _apply_search_filters(
            doc,
            entry_type=entry_type,
            topic=topic,
            min_importance=min_importance,
        ):
            continue
        importance = float(doc.get("importance", 0.5) or 0.5)
        boosted = _apply_importance_to_score(float(raw_score), importance)
        scored.append((doc, float(raw_score), boosted))

    ranked = sorted(scored, key=lambda item: item[2], reverse=True)[: max(1, max_results)]
    results: list[dict[str, Any]] = []
    for doc, raw_score, boosted in ranked:
        row = {**doc, "score": boosted}
        if explain:
            importance = float(doc.get("importance", 0.5) or 0.5)
            row["bm25_score"] = raw_score
            row["importance_boost"] = _importance_boost(importance)
        results.append(row)
    return results


def prefers_bm25_query(query: str) -> bool:
    if re.search(r"\b[0-9a-f]{7,40}\b", query, re.I):
        return True
    if re.search(r"(/|\\)[\w./-]+", query):
        return True
    if re.search(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        query,
        re.I,
    ):
        return True
    return False
