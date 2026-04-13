import os
import sys
import math
import asyncio
import logging
import tempfile
from datetime import timedelta
from typing import Callable

from telegram import Message, Document, Audio, Update, Video
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from context import is_sender_guest
from util import log, get_msg_arg, escape, reply_text

MAX_VOICE_SIZE = 1 << 20
MIN_BITRATE_K = 4
MAX_BITRATE_K = 192
QUALITY_THRESHOLD_K = 32


async def encode_opus(
    src: str,
    bitrate_k: int,
    report: Callable[[int, str], None],
    curr_len: int = 0,
) -> bytes:
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

    desc = ' '.join(args)
    report(1, f'ffmpeg: `{escape(desc)}` @ {curr_len}')
    attrs.append(os.path.basename(src))

    settings += f'({",".join(attrs)})'
    log.debug('Running ffmpeg with %s', settings)

    proc = await asyncio.create_subprocess_exec(
        'ffmpeg',
        '-hide_banner',
        '-loglevel',
        'error',
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
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    return await finalize(proc, f'ffmpeg with {settings}')


async def probe_duration(src: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        'ffprobe',
        '-v',
        'error',
        '-show_entries',
        'format=duration',
        '-of',
        'default=noprint_wrappers=1:nokey=1',
        src,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    out = await finalize(proc, 'ffprobe')
    return out.decode(errors='replace').strip()


async def finalize(proc: asyncio.subprocess.Process, desc: str) -> bytes:
    out, err = await proc.communicate()

    if err and log.isEnabledFor(logging.DEBUG):
        sys.stderr.buffer.write(err)
        sys.stderr.buffer.flush()

    ret = proc.returncode
    log.info('%s finished with %s, output %s bytes', desc, ret, len(out))

    if ret:
        lines = err.decode(errors='replace').strip().splitlines()
        msg = lines[-1] if lines else None
        raise RuntimeError(f'{desc} failed: {msg}')

    return out


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
        out = await encode_opus(src, bitrate_k, report, curr_len)
        curr_len = len(out)
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
                # new_out = await encode_opus(src, bitrate_k, report=report)
                new_result = await do_encode('Step-up', bitrate_k)
                if new_result is None:
                    break
                else:
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


type Media = Document | Audio | Video


# https://github.com/yagop/node-telegram-bot-api/issues/544
# https://openclaw.turtleand.com/topics/telegram-voice-speed-control/
async def convert_voice(
    msg: Message,
    attachment: Media,
    raw_duration: timedelta | int,
    bitrate_k: int = 0,
    quality: bool = False,
) -> None:
    log.info('Attachment: %s', attachment)
    file_name = attachment.file_name
    duration = 0

    if isinstance(raw_duration, timedelta):
        duration = raw_duration.total_seconds()
    else:
        duration = raw_duration

    attrs = [f'{attachment.file_size} bytes']
    if duration > 0:
        attrs.append(f'{duration} s')
    if bitrate_k > 0:
        attrs.append(f'{bitrate_k}k')
    if quality:
        attrs.append('quality')

    info = f'Processing: {file_name} ({", ".join(attrs)})'
    log.info('%s', info)

    status: Message | None = None
    settings: list[str | None] = [escape(info)]
    queue = asyncio.Queue()

    async def _refresh():
        nonlocal status

        if settings:
            text = '\n'.join(s for s in settings if s)
            parse_mode = 'MarkdownV2'
        else:
            text = info
            parse_mode = None

        # This needs to be serialized with a queue.
        if status is None:
            status = await msg.reply_text(text, parse_mode, do_quote=True)
        else:
            await status.edit_text(text, parse_mode)

    async def worker():
        while True:
            r = await queue.get()
            if r is None:
                return

            if isinstance(r, tuple):
                duration, data, _ = r
                await msg.reply_voice(
                    data,
                    duration=math.ceil(duration) if duration >= 0 else None,
                    do_quote=True,
                )
                return

            log.debug('Refreshing status: %s, %s', queue.qsize(), len(settings))
            await _refresh()
            await msg.reply_chat_action(ChatAction.RECORD_VOICE)

    def report(idx: int, text: str):
        idx += 1
        while len(settings) <= idx:
            settings.append(None)

        log.debug('Settings %s: %s -> %s', idx, settings[idx], text)
        settings[idx] = text

        if queue.empty():
            queue.put_nowait(True)

    asyncio.create_task(worker())

    try:
        # Report initial status before downloading the file.
        queue.put_nowait(True)

        file = await attachment.get_file()
        log.debug('File: %s', file)

        # https://github.com/aiogram/telegram-bot-api/issues/30
        file_path = file.file_path
        if file_path and os.path.isfile(file_path):
            log.debug('Using local file: %s', file_path)
            result = await encode_voice(file_path, report, duration, bitrate_k, quality)
        else:
            if s := (file_name or file_path):
                suffix = os.path.splitext(s)[1]
            else:
                suffix = None

            with tempfile.NamedTemporaryFile(prefix='voice-', suffix=suffix) as tmp:
                src = await file.download_to_drive(custom_path=tmp.name)
                result = await encode_voice(
                    str(src), report, duration, bitrate_k, quality
                )

        report(2, f'Encoded into {len(result[1])} bytes at {result[2]} kbps')
        queue.put_nowait(result)
    finally:
        queue.put_nowait(None)


def extract_media(
    msg: Message,
) -> tuple[Media, int | timedelta] | None:
    if media := msg.document:
        return media, 0
    if media := (msg.audio or msg.video):
        return media, media.duration
    return None


async def try_handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> bool:
    msg, arg = get_msg_arg(update)
    info = extract_media(msg) or (
        msg.reply_to_message and extract_media(msg.reply_to_message)
    )
    if not info:
        return False

    if not is_sender_guest():
        quality = True
    elif arg:
        quality = 'q' in arg
    else:
        quality = False

    if (p := arg.find('k')) > 0 and arg[:p].isdigit():
        bitrate_k = int(arg[:p])
    else:
        bitrate_k = 0

    await convert_voice(msg, *info, bitrate_k, quality)
    return True


async def handle_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await try_handle_voice(update, ctx):
        await reply_text(
            update,
            r'Send or reply to a media message with `/voice [q]`\.',
            'MarkdownV2',
        )
