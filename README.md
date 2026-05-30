# siftmem

Append-only JSONL memory store for AI agents and workflows. Siftmem provides type-aware indexing, BM25 keyword search, and optional Gemini-assisted importance scoring, consolidation, and session capture.

## Install

Clone or copy this directory, then install dependencies:

```bash
pip install -r requirements.txt
```

Gemini-powered features (`--score-assist`, consolidation, session capture) require a `GEMINI_API_KEY` (see below).

## Environment variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `SIFTMEM_MEMORY_DIR` | No | `~/.siftmem/memory` | Canonical JSONL store |
| `GEMINI_API_KEY` | For Gemini features | — | Importance scoring, consolidation, session capture |
| `SIFTMEM_HOME` | No | `~/.siftmem` | Base dir for session capture (`agents/main/sessions`) |

## Usage

```bash
export SIFTMEM_MEMORY_DIR=~/.siftmem/memory
export GEMINI_API_KEY=your-key   # optional unless using --score-assist / consolidate / capture

# Append a memory
python3 siftmem_append.py --type decision --topic my-topic \
  --content "Always validate inputs before indexing." --importance 0.9

# Build markdown + BM25 index
python3 siftmem_build_index.py

# Keyword search
python3 siftmem_search.py "workflow-email-triage" --json
```

With `pip install siftmem` (see packaging below):

```bash
siftmem-append --type fact --topic my-topic --content "Example fact." --importance 0.7
siftmem-build-index
siftmem-search "my-topic" --json
```

## File layout

| File / path | Role |
|-------------|------|
| `siftmem_lib.py` | Shared library (load, dedup, BM25, Gemini helpers) |
| `siftmem_append.py` | Append entries to canonical JSONL files |
| `siftmem_build_index.py` | Build markdown retrieval index + BM25 sidecar |
| `siftmem_search.py` | BM25 keyword search over JSONL corpus |
| `siftmem_consolidate.py` | Weekly topic synthesis (Gemini) |
| `siftmem_session_capture.py` | Extract memories from agent session transcripts |
| `facts.jsonl`, `decisions.jsonl`, etc. | Append-only canonical store (under `SIFTMEM_MEMORY_DIR`) |
| `siftmem_index/` | Generated markdown index (`topic__*.md`, `SIFTMEM_INDEX.md`) |
| `siftmem_bm25_index.json` | Generated BM25 search corpus |

## Memory types and importance floors

| Type | Index floor |
|------|-------------|
| decision | 0.85 |
| preference | 0.80 |
| lesson | 0.70 |
| fact | 0.60 |

Entries below the floor for their type may be excluded from the markdown index unless they are the only entry for a topic (fallback indexing).

## License

See repository root for license terms.
