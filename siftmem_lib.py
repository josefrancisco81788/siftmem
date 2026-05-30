#!/usr/bin/env python3
"""Shared Siftmem utilities."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
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


def _resolve_gemini_api_key() -> str | None:
    value = os.environ.get("GEMINI_API_KEY", "").strip()
    return value or None


def gemini_generate_json(prompt: str, user_content: str, *, model: str = "gemini-2.5-flash") -> Any | None:
    api_key = _resolve_gemini_api_key()
    if not api_key:
        return None
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{prompt}\n\n---\n\n{user_content}"}],
            }
        ],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None
    candidates = payload.get("candidates") or []
    if not candidates:
        return None
    parts = (candidates[0].get("content") or {}).get("parts") or []
    texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
    text = "\n".join(t for t in texts if t).strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def suggest_importance(entry_type: str, content: str, topic: str) -> float:
    result = gemini_generate_json(
        "Suggest a single importance score between 0 and 1 for this memory entry. "
        "Respond with JSON: {\"importance\": 0.75}",
        json.dumps({"type": entry_type, "topic": topic, "content": content}),
    )
    if isinstance(result, dict) and "importance" in result:
        try:
            return float(max(0.0, min(1.0, float(result["importance"]))))
        except (TypeError, ValueError):
            pass
    floor = importance_floor(entry_type)
    return min(1.0, floor + 0.05)


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


def build_bm25_index(memory_dir: Path = DEFAULT_MEMORY_DIR) -> dict[str, Any]:
    records = load_jsonl_records(memory_dir, resolve_supersession=True, include_superseded=False)
    if not records:
        payload = {
            "generated_at_utc": utc_now_z(),
            "corpus_size": 0,
            "documents": [],
            "corpus_tokens": [],
            "backend": "none",
        }
        BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        BM25_INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return {"ok": True, "corpus_size": 0, "path": str(BM25_INDEX_PATH)}

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
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    BM25_INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "corpus_size": len(documents), "path": str(BM25_INDEX_PATH), "backend": backend}


def bm25_search(query: str, *, max_results: int = 5, memory_dir: Path = DEFAULT_MEMORY_DIR) -> list[dict[str, Any]]:
    if not BM25_INDEX_PATH.exists():
        build_bm25_index(memory_dir)
    if not BM25_INDEX_PATH.exists():
        return []

    payload = json.loads(BM25_INDEX_PATH.read_text(encoding="utf-8"))
    documents = payload.get("documents") or []
    corpus_tokens = payload.get("corpus_tokens") or []
    if not documents or not corpus_tokens:
        return []

    query_tokens = tokenize_for_bm25(query)
    scores: list[float] = []
    try:
        from rank_bm25 import BM25Okapi

        bm25 = BM25Okapi(corpus_tokens)
        scores = list(bm25.get_scores(query_tokens))
    except ImportError:
        scores = [_simple_token_score(query_tokens, doc_tokens) for doc_tokens in corpus_tokens]

    ranked = sorted(zip(documents, scores), key=lambda item: item[1], reverse=True)[
        : max(1, max_results)
    ]
    results = []
    for doc, score in ranked:
        if score <= 0:
            continue
        results.append({**doc, "score": float(score)})
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
