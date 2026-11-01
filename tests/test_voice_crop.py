import pytest

from inoue.ffmpeg import _normalize_crop
from inoue.voice import CROP_RE, VoiceArgs, parse_time, parse_voice_args


def test_parse_time_seconds():
    assert parse_time('12') == 12
    assert parse_time('12.5') == 12.5
    assert parse_time('0') == 0


def test_parse_time_minutes_seconds():
    assert parse_time('1:30') == 90
    assert parse_time('02:15.5') == 135.5
    assert parse_time('00:45') == 45


def test_parse_time_hours_minutes_seconds():
    assert parse_time('1:02:03') == 3723
    assert parse_time('00:00:10.5') == 10.5


def test_parse_time_invalid():
    with pytest.raises(ValueError, match='could not convert string to float'):
        parse_time('invalid')


@pytest.mark.parametrize(
    ('text', 'start', 'end'),
    [
        ('10-20', '10', '20'),
        ('1:30-2:15', '1:30', '2:15'),
        ('10-', '10', None),
        ('-20', None, '20'),
    ],
)
def test_crop_re_captures_both_ends(text: str, start: str | None, end: str | None):
    m = CROP_RE.match(text)
    assert m is not None
    assert m.groups() == (start, end)


def test_parse_voice_args_reads_the_interval():
    expected = VoiceArgs(start_time=10, end_time=20, bitrate_k=24, flags='q')
    assert parse_voice_args('24kq 10-20') == expected
    assert parse_voice_args('1:30- s') == VoiceArgs(start_time=90, flags='s')
    assert parse_voice_args('q -45') == VoiceArgs(end_time=45, flags='q')


def test_parse_voice_args_rejects_a_bad_interval():
    assert parse_voice_args('24k abc-def') is None


def test_normalize_crop_bounds():
    assert _normalize_crop(10, None, None) == (10, None, None)
    assert _normalize_crop(20, 5, 15) == (10, 5, 10)
    assert _normalize_crop(20, 15, 5) == (10, 5, 10)
    assert _normalize_crop(20, -5, 30) == (20, None, None)


def test_normalize_crop_stretches_to_one_second():
    assert _normalize_crop(0.5, 0.1, 0.4) == (0.5, None, None)
    assert _normalize_crop(10, 9.8, None) == (1, 9, None)
    assert _normalize_crop(10, 5.8, None) == (4.2, 5.8, None)
    assert _normalize_crop(10, 5.8, 6) == (1, 5.8, 1)
    assert _normalize_crop(10, None, 0.5) == (1, None, 1)
