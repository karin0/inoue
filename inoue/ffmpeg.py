import asyncio
import os
import sys

from typing import TYPE_CHECKING, Literal, NamedTuple, overload

from bot import escape

from .drum import detect_bpm_and_offset, generate_drum_track
from .log import is_debug, log

if TYPE_CHECKING:
    from collections.abc import Callable


@overload
async def run_ffmpeg(
    *args: str, desc: str = '', prog: str = 'ffmpeg', capture: Literal[False] = False
) -> None: ...


@overload
async def run_ffmpeg(
    *args: str, desc: str = '', prog: str = 'ffmpeg', capture: Literal[True] = True
) -> bytes: ...


async def run_ffmpeg(
    *args: str, desc: str = '', prog: str = 'ffmpeg', capture: bool = False
) -> bytes | None:
    proc = await asyncio.create_subprocess_exec(
        prog,
        '-hide_banner',
        *(() if is_debug else ('-v', 'warning')),
        *args,
        stdout=asyncio.subprocess.PIPE if capture else None,
        stderr=None if is_debug else asyncio.subprocess.PIPE,
    )

    out, err = await proc.communicate()

    if err:
        sys.stderr.buffer.write(err)
        sys.stderr.buffer.flush()

    ret = proc.returncode
    desc = f'{prog} ({desc})' if desc else prog

    if capture:
        log.info('%s finished with %s, output %s bytes', desc, ret, len(out))
    else:
        log.info('%s finished with %s', desc, ret)

    if ret:
        if err and (text := err.decode(errors='replace').strip()):
            msg = text[text.rfind('\n') + 1 :]
        else:
            msg = None
        raise RuntimeError(f'{desc} failed: {msg}')

    return out


MAX_VOICE_SIZE = 1 << 20
MIN_BITRATE_K = 4
MAX_BITRATE_K = 192
QUALITY_THRESHOLD_K = 32


async def _encode_opus(
    src: str,
    bitrate_k: int,
    settings: str,
    start_time: float | None,
    duration: float | None,
    speed: float,
    drum_path: str | None,
) -> tuple[bytes, str]:
    args = ['-b:a', f'{bitrate_k}k']

    if bitrate_k > QUALITY_THRESHOLD_K:
        args += ('-frame_duration', '60')
    else:
        args += ('-frame_duration', '120', '-ac', '1', '-application', 'voip')

    inputs = ['-i', src]

    if speed != 1:
        a0_label = 'a_speed'
        filter_chunks = [
            f'[0:a:0]aresample=44100,asetrate=44100*{speed},aresample=44100[{a0_label}]'
        ]
    else:
        a0_label = 'a0'
        filter_chunks = [f'[0:a:0]aresample=44100[{a0_label}]']

    if drum_path:
        inputs += ('-i', drum_path)
        filter_chunks.append('[1:a:0]volume=1[a1]')
        filter_chunks.append(
            f'[{a0_label}][a1]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[out]'
        )
        filters = ('-filter_complex', ';'.join(filter_chunks))
        maps = ('-map', '[out]')
    elif speed != 1.0:
        filters = (
            '-filter_complex',
            f'[0:a:0]aresample=44100,asetrate=44100*{speed},aresample=44100[out]',
        )
        maps = ('-map', '[out]')
    else:
        filters = ()
        maps = ('-map', '0:a:0')

    out = await run_ffmpeg(
        *(('-ss', str(start_time)) if start_time is not None else ()),
        *inputs,
        *(('-t', str(duration)) if duration is not None else []),
        *filters,
        *maps,
        '-vn',
        '-sn',
        '-dn',
        '-map_metadata',
        '-1',
        '-c:a',
        'libopus',
        *args,
        '-vbr',
        'on',
        '-compression_level',
        '10',
        '-f',
        'ogg',
        'pipe:1',
        desc=f'voice/{settings}',
        capture=True,
    )
    return out, ' '.join(args)


async def encode_opus(
    src: str,
    bitrate_k: int,
    start_time: float | None,
    duration: float | None,
    full_duration: float,
    speed: float,
    drum: tuple[float, float] | None,
) -> tuple[bytes, str]:
    settings = f'{bitrate_k}k'
    attrs = []
    if speed != 1:
        attrs.append(f'speed:{speed}')
    if drum is not None:
        bpm, offset = drum
        attrs.append(f'drum:{bpm:.1f}')

    attrs.append(os.path.basename(src))

    if start_time is not None or duration is not None:
        attrs.append(f'crop:{start_time or 0}+{duration or ''}')

    settings += f'({','.join(attrs)})'
    log.debug('Running ffmpeg with %s', settings)

    drum_path = None
    if drum is not None:
        import tempfile

        bpm, offset = drum

        fd, drum_path = tempfile.mkstemp(suffix='.wav')
        try:
            os.close(fd)
            drum_bpm = bpm * speed
            offset_new = offset / speed
            drum_dur = full_duration / speed
            if drum_dur <= 0:
                drum_dur = 30
            generate_drum_track(drum_path, drum_dur, drum_bpm, offset_new)
            return await _encode_opus(
                src, bitrate_k, settings, start_time, duration, speed, drum_path
            )
        finally:
            try:
                os.remove(drum_path)
            except OSError:
                log.exception('Failed to remove drum track: %s', drum_path)

    return await _encode_opus(src, bitrate_k, settings, start_time, duration, speed, None)


async def probe_duration(src: str) -> str:
    out = await run_ffmpeg(
        '-show_entries',
        'format=duration',
        '-of',
        'default=noprint_wrappers=1:nokey=1',
        src,
        prog='ffprobe',
        capture=True,
    )
    return out.decode(errors='replace').strip()


def _clamp_bitrate(bitrate_k: int) -> int:
    return max(MIN_BITRATE_K, min(MAX_BITRATE_K, bitrate_k))


def _estimate_bitrate_from_duration(duration: float, quality: bool) -> int:
    duration *= 117 if quality else 125
    raw_budget_k = round(MAX_VOICE_SIZE / duration)
    return _clamp_bitrate(raw_budget_k)


def _estimate_bitrate_from_sample(sample_bitrate_k: int, sample_size: int) -> int:
    if sample_size <= 0:
        return MIN_BITRATE_K
    estimated = int(sample_bitrate_k * MAX_VOICE_SIZE / sample_size)
    if estimated >= sample_bitrate_k:
        estimated = sample_bitrate_k - 1
    return _clamp_bitrate(estimated)


def _normalize_crop(
    duration: float, start_time: float | None, end_time: float | None
) -> tuple[float, float | None, float | None]:
    s = min(max(0, start_time), duration) if start_time is not None else 0
    e = min(max(0, end_time), duration) if end_time is not None else duration
    if s > e:
        s, e = e, s

    if (dur := e - s) < 1:
        if duration < 1:
            s = 0
            e = dur = duration
        elif (e := s + (dur := 1)) > duration:
            s = duration - 1
            e = duration

    return dur, s or None, (None if e == duration else dur)


class EncodedVoice(NamedTuple):
    duration: float
    data: bytes
    bitrate_k: int
    iterations: int


async def encode_voice(
    src: str,
    report: Callable[[int, str], None],
    duration: float,
    bitrate_k: int = 0,
    quality: bool = False,
    start_time: float | None = None,
    end_time: float | None = None,
    speed: float = 1,
    drum: bool = False,
) -> EncodedVoice:
    curr_len = 0
    raw_result: EncodedVoice | None = None
    iterations = 0

    if duration <= 0:
        value = await probe_duration(src)
        log.info('ffprobe: %s', value)
        report(0, f'ffprobe: `{escape(value)}`')
        duration = float(value)
        if duration <= 0:
            raise ValueError(f'Invalid duration: {duration}')

    duration, start_time, crop_duration = _normalize_crop(duration, start_time, end_time)
    log.info('Encoding voice: crop=%s+%s duration=%.1f', start_time, crop_duration, duration)

    adj_duration = duration / speed if speed != 1 else duration

    drum_grid = None
    if drum:
        try:
            drum_grid = await detect_bpm_and_offset(src, start_time, crop_duration)
        except (RuntimeError, OSError, ValueError) as e:
            log.warning('Failed to detect bpm: %s', e)
            report(0, 'bpm: `failed`')
        else:
            report(0, f'bpm: `{drum_grid[0]:.1f}`')

    async def do_encode(desc: str, bitrate_k: int) -> EncodedVoice | None:
        nonlocal curr_len, raw_result, iterations
        out, info = await encode_opus(
            src, bitrate_k, start_time, crop_duration, duration, speed, drum_grid
        )
        curr_len = len(out)
        report(1, f'ffmpeg: `{info}` @ {curr_len}')
        success = curr_len <= MAX_VOICE_SIZE
        iterations += 1
        log.info(
            '[%d] %s encode %s: duration=%s bitrate=%sk output=%s',
            iterations,
            desc,
            'success' if success else 'failed',
            adj_duration,
            bitrate_k,
            curr_len,
        )
        raw_result = EncodedVoice(adj_duration, out, bitrate_k, iterations)
        if success:
            return raw_result

    if bitrate_k > 0 and (result := await do_encode('Hinted', bitrate_k)):
        return result

    bitrate_k = _estimate_bitrate_from_duration(adj_duration, quality)
    result = await do_encode('Duration-based', bitrate_k)

    end = min(bitrate_k + 10, MAX_BITRATE_K)
    if result is None:
        inferred_bitrate_k = _estimate_bitrate_from_sample(bitrate_k, curr_len)
        if inferred_bitrate_k < bitrate_k:
            end = min(inferred_bitrate_k + 10, bitrate_k - 1)
            bitrate_k = inferred_bitrate_k
            result = await do_encode('Sample-based', inferred_bitrate_k)

    if result is not None:
        if quality:
            for bitrate_k in range(bitrate_k + 1, end + 1):
                new_result = await do_encode('Step-up', bitrate_k)
                if new_result is None:
                    return result
                result = new_result
        return result

    # Oversized output: decrease bitrate 1 kbps at a time to maximize quality.
    end = max(MIN_BITRATE_K, bitrate_k - 10)
    for bitrate_k in range(bitrate_k - 1, end - 1, -1):
        if (result := await do_encode('Step-down', bitrate_k)) is not None:
            return result

    # Final fallback: return the latest output.
    assert raw_result is not None
    log.warning(
        'Voice still exceeds 1 MiB at bitrate=%sk, output=%s bytes', raw_result.bitrate_k, curr_len
    )
    return raw_result


VIDEO_NOTE_MAX_DURATION = 60
VIDEO_NOTE_SIDE = 640

# Telegram rejects video notes larger than 12 MiB.
VIDEO_NOTE_MAX_BYTES = 12 << 20
VIDEO_NOTE_AUDIO_BITRATE_K = 64


async def encode_video_note(src: str, duration: float) -> str:
    dst = os.path.splitext(src)[0] + '.note.mp4'
    if os.path.isfile(dst):
        log.info('Cached video note: %s', dst)
        return dst

    duration = min(max(duration, 1.0), VIDEO_NOTE_MAX_DURATION)

    # Reserve 512 KiB for container/muxing overhead and rate control slack.
    budget_bits = (VIDEO_NOTE_MAX_BYTES - (512 << 10)) << 3
    total_bitrate_k = int(budget_bits / (duration * 1000))
    video_bitrate_k = max(200, total_bitrate_k - VIDEO_NOTE_AUDIO_BITRATE_K)

    log.info('Encoding video note: %s (%.1fs, %dk)', dst, duration, video_bitrate_k)
    await run_ffmpeg(
        '-i',
        src,
        '-t',
        str(VIDEO_NOTE_MAX_DURATION),
        '-vf',
        f"crop='min(iw,ih)':'min(iw,ih)',scale={VIDEO_NOTE_SIDE}:{VIDEO_NOTE_SIDE}",
        '-c:v',
        'libx264',
        '-preset',
        'veryfast',
        '-b:v',
        f'{video_bitrate_k}k',
        '-maxrate',
        f'{video_bitrate_k}k',
        '-bufsize',
        f'{video_bitrate_k * 2}k',
        '-pix_fmt',
        'yuv420p',
        '-c:a',
        'aac',
        '-b:a',
        f'{VIDEO_NOTE_AUDIO_BITRATE_K}k',
        '-movflags',
        '+faststart',
        dst,
        desc=f'video note/{video_bitrate_k}k',
    )

    log.info('Video note encoded into %d bytes', os.path.getsize(dst))
    return dst
