# siftmem

Append-only JSONL memory store for AI agents and workflows. Siftmem provides type-aware indexing, BM25 keyword search, and optional LLM-assisted importance scoring, consolidation, and session capture.

**Core features work fully offline** — no API key required for append, index, search, or dedup.

## Install

```bash
pip install siftmem
```

From source:

```bash
git clone https://github.com/josefrancisco81788/siftmem.git
cd siftmem
pip install -e .
```

Optional extras:

```bash
pip install siftmem[llm-openai]   # OpenAI provider for capture/consolidate/score-assist
pip install siftmem[dev]          # pytest, build, twine
```

## 5-minute quickstart

```bash
pip install siftmem
siftmem-init
siftmem-search "getting-started" --json
siftmem-append --type fact --topic my-topic --content "Your first memory."
siftmem-build-index
siftmem-doctor
```

`--importance` is optional; siftmem defaults to the type floor plus a small margin.

## Works offline vs needs LLM

| Feature | API key required? |
|---------|-------------------|
| `siftmem-append` (default importance) | No |
| `siftmem-build-index` | No |
| `siftmem-search` | No |
| Dedup / supersession | No |
| `siftmem-init`, `siftmem-doctor` | No |
| `--score-assist` | Yes (when `SIFTMEM_LLM_PROVIDER` is set) |
| `siftmem-consolidate` | Yes |
| `siftmem-capture` | Yes |

## Environment variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `SIFTMEM_MEMORY_DIR` | No | `~/.siftmem/memory` | Canonical JSONL store |
| `SIFTMEM_LLM_PROVIDER` | No | `none` | LLM backend: `none`, `gemini`, `openai` |
| `SIFTMEM_LLM_MODEL` | No | provider default | Model override |
| `GEMINI_API_KEY` | For `gemini` provider | — | Gemini API access |
| `OPENAI_API_KEY` | For `openai` provider | — | OpenAI API access |
| `SIFTMEM_HOME` | No | `~/.siftmem` | Base dir for session capture (`agents/main/sessions`) |

## Usage

```bash
export SIFTMEM_MEMORY_DIR=~/.siftmem/memory

# Append a memory (importance optional)
siftmem-append --type decision --topic my-topic \
  --content "Always validate inputs before indexing."

# Optional LLM-refined importance
export SIFTMEM_LLM_PROVIDER=gemini
export GEMINI_API_KEY=your-key
siftmem-append --type decision --topic my-topic \
  --content "..." --score-assist

# Build markdown + BM25 index
siftmem-build-index

# Keyword search with filters
siftmem-search "workflow-email-triage" --type decision --json
siftmem-search "paths /tmp/foo" --explain --json
```

## Python API

```python
from pathlib import Path
from siftmem import MemoryStore

store = MemoryStore(Path("~/.siftmem/memory"))
store.append(
    type="decision",
    topic="onboarding",
    content="Always rebuild the index after batch writes.",
    importance=0.9,
    rebuild_index=True,
)
hits = store.search("onboarding", max_results=5)
print(store.stats())
```

## Package layout

| Module | Role |
|--------|------|
| `siftmem.store` | `MemoryStore` Python API |
| `siftmem.lib` | Shared library (load, dedup, BM25) |
| `siftmem.llm` | Pluggable LLM JSON generation |
| `siftmem.append` | Append entries to canonical JSONL files |
| `siftmem.build_index` | Build markdown retrieval index + BM25 sidecar |
| `siftmem.search` | BM25 keyword search over JSONL corpus |
| `siftmem.init_cmd` | Bootstrap a new memory store |
| `siftmem.doctor` | Health checks |
| `siftmem.consolidate` | Weekly topic synthesis (LLM) |
| `siftmem.session_capture` | Extract memories from agent session transcripts |

## Generated artifacts (under `SIFTMEM_MEMORY_DIR`)

| Path | Role |
|------|------|
| `facts.jsonl`, `decisions.jsonl`, etc. | Append-only canonical store |
| `siftmem_index/` | Markdown index (`topic__*.md`, `SIFTMEM_INDEX.md`) |
| `siftmem_bm25_index.json` | BM25 search corpus |

## Memory types and importance floors

| Type | Index floor | Default append |
|------|-------------|----------------|
| decision | 0.85 | 0.90 |
| preference | 0.80 | 0.85 |
| lesson | 0.70 | 0.75 |
| fact | 0.60 | 0.65 |

Entries below the floor for their type may be excluded from the markdown index unless they are the only entry for a topic (fallback indexing).

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

## License

MIT — see [LICENSE](LICENSE).
