import os
import sys
import asyncio
import logging
from datetime import timedelta
from typing import Callable

from telegram import Bot, Message, Document, Audio
from telegram.constants import ChatAction

from util import log, escape

MAX_VOICE_SIZE = 1 << 20
MIN_BITRATE_K = 16
MAX_BITRATE_K = 48
BEST_BITRATE_K = 64
QUALITY_THRESHOLD_K = 32


async def run_ffmpeg(
    raw: bytearray,
    bitrate_k: int,
    report: Callable[[str], None],
) -> bytes:
    args = ['-b:a', f'{bitrate_k}k']
    quality = bitrate_k > QUALITY_THRESHOLD_K

    if quality:
        # Standard quality settings.
        args += ('-frame_duration', '60')
    else:
        args += ('-frame_duration', '120', '-ac', '1', '-application', 'voip')

    settings = f'{bitrate_k}k'
    if quality:
        settings += '(q)'

    report(' '.join(args))

    log.debug('Running ffmpeg with %s', settings)
    proc = await asyncio.create_subprocess_exec(
        'ffmpeg',
        '-hide_banner',
        '-loglevel',
        'error',
        '-i',
        'pipe:0',
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
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    out, err = await proc.communicate(raw)

    if err and log.isEnabledFor(logging.DEBUG):
        sys.stderr.buffer.write(err)
        sys.stderr.buffer.flush()

    log.info(
        'ffmpeg with %s finished with %s, output %s bytes',
        settings,
        proc.returncode,
        len(out),
    )

    if proc.returncode != 0:
        msg = (
            err.decode(errors='replace').strip().splitlines()[-1:] or ['unknown error']
        )[0]
        raise RuntimeError(f'ffmpeg failed: {msg}')

    return out


def _clamp_bitrate(bitrate_k: int) -> int:
    return max(MIN_BITRATE_K, min(MAX_BITRATE_K, bitrate_k))


def _estimate_bitrate_from_duration(duration: int) -> int:
    raw_budget_k = int(MAX_VOICE_SIZE * 8 / duration / 1000)
    return _clamp_bitrate(raw_budget_k)


def _estimate_bitrate_from_sample(sample_bitrate_k: int, sample_size: int) -> int:
    if sample_size <= 0:
        return MIN_BITRATE_K
    estimated = int(sample_bitrate_k * MAX_VOICE_SIZE / sample_size)
    if estimated >= sample_bitrate_k:
        estimated = sample_bitrate_k - 1
    return _clamp_bitrate(estimated)


async def encode_voice(
    raw: bytearray,
    report: Callable[[str], None],
    duration: int,
) -> bytes:
    bitrate_k = (
        _estimate_bitrate_from_duration(duration) if duration else BEST_BITRATE_K
    )
    out = await run_ffmpeg(raw, bitrate_k, report=report)
    log.info(
        'Initial encode: duration=%s target=%sk output=%s bytes',
        duration,
        bitrate_k,
        len(out),
    )

    if len(out) <= MAX_VOICE_SIZE:
        return out

    if duration:
        inferred_bitrate_k = _estimate_bitrate_from_sample(bitrate_k, len(out))
        if inferred_bitrate_k < bitrate_k:
            bitrate_k = inferred_bitrate_k
            out = await run_ffmpeg(raw, bitrate_k, report=report)
            log.info(
                'Sample-based estimate: probe=%sk inferred=%sk output=%s bytes',
                BEST_BITRATE_K,
                bitrate_k,
                len(out),
            )
            if len(out) <= MAX_VOICE_SIZE:
                return out

    # Oversized output: decrease bitrate 1 kbps at a time to maximize quality.
    while bitrate_k > MIN_BITRATE_K:
        bitrate_k -= 1
        out = await run_ffmpeg(raw, bitrate_k, report=report)
        if len(out) <= MAX_VOICE_SIZE:
            log.info(
                'Step-down encode success: bitrate=%sk output=%s bytes',
                bitrate_k,
                len(out),
            )
            return out

    # Final fallback: return the latest output.
    log.warning(
        'Voice still exceeds 1 MiB at min bitrate=%sk, output=%s bytes',
        bitrate_k,
        len(out),
    )
    return out


# https://github.com/yagop/node-telegram-bot-api/issues/544
# https://openclaw.turtleand.com/topics/telegram-voice-speed-control/
async def handle_voice(msg: Message, bot: Bot, attachment: Document | Audio) -> None:
    log.info('Attachment: %s', attachment)
    file_name = attachment.file_name
    if duration := msg.audio and msg.audio.duration:
        if isinstance(duration, timedelta):
            duration = int(duration.total_seconds())
    else:
        duration = 0

    info = f'Processing: {file_name} ({attachment.file_size} bytes, {duration} secs)'
    log.info('%s', info)

    status: Message | None = None
    settings: str | None = None

    async def refresh_status():
        nonlocal status
        if settings is None:
            text = info
            parse_mode = None
        else:
            text = f'{escape(info)}\nffmpeg: `{escape(settings)}`'
            parse_mode = 'MarkdownV2'

        if status is None:
            status = await msg.reply_text(text, parse_mode, do_quote=True)
        else:
            await status.edit_text(text, parse_mode)

        await msg.reply_chat_action(ChatAction.RECORD_VOICE)

    def report(args: str):
        nonlocal settings
        log.debug('Settings: %s -> %s', settings, args)
        settings = args
        asyncio.create_task(refresh_status())

    # Report initial status before downloading the file.
    asyncio.create_task(refresh_status())

    file = await bot.get_file(attachment.file_id)
    raw = await file.download_as_bytearray()

    voice = await encode_voice(raw, report, duration)
    await msg.reply_voice(voice, duration=duration, do_quote=True)
    return
