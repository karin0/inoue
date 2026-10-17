from typing import TYPE_CHECKING

import pytest

from inoue import ytdlp
from inoue.ytdlp import get_ytdlp

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def fresh_instances(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ytdlp, '_instances', [None, None])


def test_downloads_retry():
    params = get_ytdlp(False).params
    assert params.get('retries')
    assert params.get('fragment_retries')
