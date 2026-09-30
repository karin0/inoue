import array
import math
import wave

from typing import TYPE_CHECKING

import pytest

from inoue.drum import detect_bpm_and_offset, generate_drum_track
from inoue.voice import extract_effects

if TYPE_CHECKING:
    from pathlib import Path


def test_extract_effects_enables_nightcore():
    assert extract_effects('nc') == ('', 1.5, False)
    assert extract_effects('nightcore') == ('', 1.5, False)
    assert extract_effects('NC') == ('NC', 1, False)
    assert extract_effects('NIGHTCORE') == ('NIGHTCORE', 1, False)


def test_extract_effects_reads_speed():
    assert extract_effects('nc1.3') == ('', 1.3, False)
    assert extract_effects('nc=1.4') == ('', 1.4, False)
    assert extract_effects('nightcore1.25') == ('', 1.25, False)
    assert extract_effects('nightcore=1.1') == ('', 1.1, False)


def test_extract_effects_keeps_other_arguments():
    assert extract_effects('nc1.4 24kq') == ('24kq', 1.4, False)
    assert extract_effects('24k nc s') == ('24k s', 1.5, False)
    assert extract_effects('q -45 nc1.2') == ('q -45', 1.2, False)
    assert extract_effects('abc nc_speed') == ('abc nc_speed', 1, True)


def test_extract_effects_reads_drum():
    assert extract_effects('nc d') == ('d', 1.5, True)
    assert extract_effects('nc drum') == ('drum', 1.5, True)
    assert extract_effects('nc1.2 d 24kq') == ('d 24kq', 1.2, True)
    assert extract_effects('drum') == ('drum', 1, True)
    assert extract_effects('d') == ('d', 1, True)


def test_generate_drum_track_writes_mono_pcm(tmp_path: Path):
    path = tmp_path / 'drum.wav'
    generate_drum_track(str(path), 2.5, 120, 0.2)
    with wave.open(str(path), 'rb') as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 44100
        assert w.getnframes() == 2.5 * 44100


def _write_wav(path: Path, rate: int, samples: array.array) -> None:
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


async def test_detect_bpm_and_offset_of_periodic_pulses(tmp_path: Path):
    # 0.1 s bursts of a 440 Hz tone every 0.5 s from 0.2 s on, which is 120 BPM.
    rate = 22050
    samples = array.array('h', [0] * (3 * rate))
    for beat in range(6):
        start = int((0.2 + beat * 0.5) * rate)
        for j in range(int(0.1 * rate)):
            samples[start + j] = int(20000 * math.sin(2 * math.pi * 440 * j / rate))
    path = tmp_path / 'pulses.wav'
    _write_wav(path, rate, samples)

    bpm, offset = await detect_bpm_and_offset(str(path), start_time=None, duration=None)
    assert bpm == pytest.approx(120, abs=3)
    assert offset == pytest.approx(0.2, abs=0.1)


async def test_detect_bpm_and_offset_rejects_missing_file(tmp_path: Path):
    with pytest.raises(RuntimeError):
        await detect_bpm_and_offset(str(tmp_path / 'missing.mp3'), start_time=None, duration=None)


async def test_detect_bpm_and_offset_rejects_short_audio(tmp_path: Path):
    path = tmp_path / 'short.wav'
    _write_wav(path, 8000, array.array('h', [0] * 100))
    with pytest.raises(ValueError, match='Not enough'):
        await detect_bpm_and_offset(str(path), start_time=None, duration=None)
