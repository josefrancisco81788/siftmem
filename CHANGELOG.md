# Changelog

## 0.2.0

- Default importance when `--importance` is omitted (no API key required).
- `siftmem-init` and `siftmem-doctor` onboarding and health commands.
- Pluggable LLM providers via `SIFTMEM_LLM_PROVIDER` (`none`, `gemini`, `openai`).
- `MemoryStore` Python API for programmatic use.
- Search filters (`--type`, `--topic`, `--min-importance`) and importance-weighted ranking.
- Pytest suite and GitHub Actions CI.

## 0.1.0

- Initial release: JSONL append, markdown index, BM25 search, optional Gemini features.
