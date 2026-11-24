from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _clean_process_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # default cache/calibration directories never touch the repo
    for var in ("DISSEQT_JUDGE_CACHE", "DISSEQT_ALLOW_JUDGE_REQUESTS", "DISSEQT_JUDGE_MODEL", "CI"):
        monkeypatch.delenv(var, raising=False)
