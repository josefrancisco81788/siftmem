# siftmem

Append-only JSONL memory store for AI agents and workflows. Siftmem provides type-aware indexing, BM25 keyword search, and optional Gemini-assisted importance scoring, consolidation, and session capture.

## Install

From PyPI (when published):

```bash
pip install siftmem
```

From source:

```bash
git clone https://github.com/josefrancisco81788/siftmem.git
cd siftmem
pip install -e .
```

Or install dependencies only:

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
siftmem-append --type decision --topic my-topic \
  --content "Always validate inputs before indexing." --importance 0.9

# Build markdown + BM25 index
siftmem-build-index

# Keyword search
siftmem-search "workflow-email-triage" --json
```

Module invocation (without console scripts):

```bash
python -m siftmem.append --type fact --topic my-topic --content "Example." --importance 0.7
python -m siftmem.build_index
python -m siftmem.search "my-topic" --json
```

## Package layout

| Module | Role |
|--------|------|
| `siftmem.lib` | Shared library (load, dedup, BM25, Gemini helpers) |
| `siftmem.append` | Append entries to canonical JSONL files |
| `siftmem.build_index` | Build markdown retrieval index + BM25 sidecar |
| `siftmem.search` | BM25 keyword search over JSONL corpus |
| `siftmem.consolidate` | Weekly topic synthesis (Gemini) |
| `siftmem.session_capture` | Extract memories from agent session transcripts |

## Generated artifacts (under `SIFTMEM_MEMORY_DIR`)

| Path | Role |
|------|------|
| `facts.jsonl`, `decisions.jsonl`, etc. | Append-only canonical store |
| `siftmem_index/` | Markdown index (`topic__*.md`, `SIFTMEM_INDEX.md`) |
| `siftmem_bm25_index.json` | BM25 search corpus |

## Memory types and importance floors

| Type | Index floor |
|------|-------------|
| decision | 0.85 |
| preference | 0.80 |
| lesson | 0.70 |
| fact | 0.60 |

Entries below the floor for their type may be excluded from the markdown index unless they are the only entry for a topic (fallback indexing).

## License

MIT
