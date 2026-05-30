#!/usr/bin/env python3
"""BM25 keyword search over canonical Siftmem JSONL corpus."""

from __future__ import annotations

import argparse
import json
import sys

from siftmem_lib import (
    DEFAULT_MEMORY_DIR,
    bm25_search,
    build_bm25_index,
    log_event,
    prefers_bm25_query,
)



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search Siftmem via BM25.")
    parser.add_argument("query", help="Search query string.")
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--memory-dir", default=str(DEFAULT_MEMORY_DIR))
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    memory_dir = __import__("pathlib").Path(args.memory_dir)

    if args.rebuild_index:
        build_bm25_index(memory_dir)

    results = bm25_search(args.query, max_results=args.max_results, memory_dir=memory_dir)
    top_score = results[0]["score"] if results else 0.0

    payload = {
        "query": args.query,
        "prefer_bm25_heuristic": prefers_bm25_query(args.query),
        "results": results,
        "count": len(results),
        "top_score": top_score,
    }

    if not args.dry_run:
        log_event(
            "search",
            memory_dir=memory_dir,
            query=args.query,
            results_returned=len(results),
            top_score=top_score,
        )

    if args.as_json or args.dry_run:
        print(json.dumps(payload, indent=2))
    else:
        for idx, row in enumerate(results, start=1):
            print(f"{idx}. [{row.get('score', 0):.3f}] {row.get('topic')} ({row.get('type')})")
            print(f"   {row.get('content', '')[:200]}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
