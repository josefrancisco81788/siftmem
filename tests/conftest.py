from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def memory_dir(tmp_path: Path) -> Path:
    store = tmp_path / "memory"
    store.mkdir()
    return store
