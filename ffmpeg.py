import os
import sys
import asyncio
from typing import Literal, Callable, overload

from util import escape, log, is_debug


@overload
async def run_ffmpeg(
    *args: str,
    desc: str = '',
    prog: str = 'ffmpeg',
    capture: Literal[False] = False,
) -> None: ...


@overload
async def run_ffmpeg(
    *args: str,
    desc: str = '',
    prog: str = 'ffmpeg',
    capture: Literal[True] = True,
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
    if not desc:
        desc = prog

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


async def encode_opus(
    src: str,
    bitrate_k: int,
) -> tuple[bytes, str]:
    args = ['-b:a', f'{bitrate_k}k']
    quality = bitrate_k > QUALITY_THRESHOLD_K

    if quality:
        # Standard quality settings.
        args += ('-frame_duration', '60')
    else:
        args += ('-frame_duration', '120', '-ac', '1', '-application', 'voip')

    settings = f'{bitrate_k}k'
    attrs = []
    if quality:
        attrs.append('q')

    attrs.append(os.path.basename(src))

    settings += f'({",".join(attrs)})'
    log.debug('Running ffmpeg with %s', settings)

    out = await run_ffmpeg(
        '-xerror',
        '-i',
        src,
        '-map',
        '0:a:0',
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
        desc=f'ffmpeg (voice/{settings})',
        capture=True,
    )
    return out, ' '.join(args)


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


async def encode_voice(
    src: str,
    report: Callable[[int, str], None],
    duration: float,
    bitrate_k: int = 0,
    quality: bool = False,
) -> tuple[float, bytes, int]:
    curr_len = 0
    raw_result: tuple[float, bytes, int] | None = None

    async def do_encode(desc: str, bitrate_k: int) -> tuple[float, bytes, int] | None:
        nonlocal curr_len, raw_result
        out, info = await encode_opus(src, bitrate_k)
        curr_len = len(out)
        report(1, f'ffmpeg: `{info}` @ {curr_len}')
        success = curr_len <= MAX_VOICE_SIZE
        log.info(
            '%s encode %s: duration=%s bitrate=%sk output=%s',
            desc,
            'success' if success else 'failed',
            duration,
            bitrate_k,
            curr_len,
        )
        raw_result = duration, out, bitrate_k
        if success:
            return raw_result

    if bitrate_k > 0 and (result := await do_encode('Hinted', bitrate_k)):
        return result

    if duration <= 0:
        value = await probe_duration(src)
        log.info('ffprobe: %s', value)
        report(0, f'ffprobe: `{escape(value)}`')
        duration = float(value)
        if duration <= 0:
            raise ValueError('Invalid duration: %s', duration)

    bitrate_k = _estimate_bitrate_from_duration(duration, quality)
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
        'Voice still exceeds 1 MiB at bitrate=%sk, output=%s bytes',
        raw_result[2],
        curr_len,
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
        '-xerror',
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
        desc=f'ffmpeg (video note/{video_bitrate_k}k)',
    )

    log.info('Video note encoded into %d bytes', os.path.getsize(dst))
    return dst
