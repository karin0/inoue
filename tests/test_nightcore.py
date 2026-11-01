import array
import math
import wave

from typing import TYPE_CHECKING

import pytest

from inoue.drum import detect_bpm_and_offset, generate_drum_track
from inoue.voice import VoiceArgs, parse_voice_args

if TYPE_CHECKING:
    from pathlib import Path


def test_parse_voice_args_enables_nightcore():
    assert parse_voice_args('nc') == VoiceArgs(speed=1.5)
    assert parse_voice_args('nightcore') == VoiceArgs(speed=1.5)


def test_parse_voice_args_reads_speed():
    assert parse_voice_args('nc1.3') == VoiceArgs(speed=1.3)
    assert parse_voice_args('nc=1.4') == VoiceArgs(speed=1.4)
    assert parse_voice_args('nightcore1.25') == VoiceArgs(speed=1.25)
    assert parse_voice_args('nightcore=1.1') == VoiceArgs(speed=1.1)


def test_parse_voice_args_reads_flags_and_bitrate():
    assert parse_voice_args('nc1.4 24kq') == VoiceArgs(speed=1.4, bitrate_k=24, flags='q')
    assert parse_voice_args('24k nc s') == VoiceArgs(speed=1.5, bitrate_k=24, flags='s')
    assert parse_voice_args('Qs 64k') == VoiceArgs(bitrate_k=64, flags='Qs')
    assert parse_voice_args('') == VoiceArgs()


def test_parse_voice_args_reads_drum():
    assert parse_voice_args('nc d') == VoiceArgs(speed=1.5, flags='d')
    assert parse_voice_args('nc drum') == VoiceArgs(speed=1.5, flags='d')
    assert parse_voice_args('nc1.2 d 24kq') == VoiceArgs(speed=1.2, bitrate_k=24, flags='dq')


@pytest.mark.parametrize('arg', ['NC', 'nc_speed', 'nc1..2', 'sad', '好听', 'nc 好听', '-', 'k'])
def test_parse_voice_args_rejects_other_words(arg: str):
    assert parse_voice_args(arg) is None


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
