from typing import TYPE_CHECKING

import pytest

from inoue import ytdlp
from inoue.ytdlp import MAX_FILE_SIZE, get_ytdlp

if TYPE_CHECKING:
    from pathlib import Path


def _format(format_id: str, ext: str, size: int, *, video: bool, audio: bool) -> dict:
    return {
        'format_id': format_id,
        'ext': ext,
        'url': 'https://example.com/' + format_id,
        'filesize': size,
        'vcodec': 'avc1' if video else 'none',
        'acodec': 'mp4a' if audio else 'none',
        'height': 1080 if size > MAX_FILE_SIZE else 360,
    }


# `formats` is sorted from worst to best, as yt-dlp sorts them before selecting.
def _select(audio_only: bool, formats: list[dict]) -> list[str]:
    selector = get_ytdlp(audio_only).format_selector
    ctx = {'formats': formats, 'has_merged_format': False, 'incomplete_formats': False}
    return [f['format_id'] for f in selector(ctx)]


@pytest.fixture(autouse=True)
def fresh_instances(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ytdlp, '_instances', [None, None])


def test_video_prefers_streams_that_fit():
    formats = [
        _format('small', 'mp4', MAX_FILE_SIZE // 4, video=True, audio=False),
        _format('big', 'mp4', MAX_FILE_SIZE * 2, video=True, audio=False),
        _format('a', 'm4a', MAX_FILE_SIZE // 8, video=False, audio=True),
    ]
    assert _select(False, formats) == ['small+a']


def test_video_fits_with_its_audio():
    formats = [
        _format('a_small', 'm4a', MAX_FILE_SIZE // 4, video=False, audio=True),
        _format('a_best', 'm4a', MAX_FILE_SIZE * 6 // 10, video=False, audio=True),
        _format('v', 'mp4', MAX_FILE_SIZE // 2, video=True, audio=False),
    ]
    assert _select(False, formats) == ['v+a_small']


def test_video_leaves_room_for_merging():
    formats = [
        _format('a_small', 'm4a', MAX_FILE_SIZE * 2 // 100, video=False, audio=True),
        _format('a_best', 'm4a', MAX_FILE_SIZE * 5 // 100, video=False, audio=True),
        _format('v', 'mp4', MAX_FILE_SIZE * 94 // 100, video=True, audio=False),
    ]
    assert _select(False, formats) == ['v+a_small']


def test_video_falls_back_to_an_oversized_stream():
    formats = [
        _format('big', 'mp4', MAX_FILE_SIZE * 2, video=True, audio=False),
        _format('a', 'm4a', MAX_FILE_SIZE // 8, video=False, audio=True),
    ]
    assert _select(False, formats) == ['big+a']


def test_audio_prefers_streams_that_fit():
    formats = [
        _format('a_big', 'm4a', MAX_FILE_SIZE * 2, video=False, audio=True),
        _format('a_small', 'webm', MAX_FILE_SIZE // 8, video=False, audio=True),
    ]
    assert _select(True, formats) == ['a_small']


def test_downloads_retry():
    params = get_ytdlp(False).params
    assert params.get('retries')
    assert params.get('fragment_retries')
