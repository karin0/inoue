import os
import math
import asyncio
from datetime import timedelta

from telegram import Message, Document, Audio, Update, Video
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from ffmpeg import encode_voice
from context import is_sender_guest
from ytdlp import run_ytdlp, extract_url, Output
from util import (
    log,
    get_msg_arg,
    escape,
    reply_text,
    keep_chat_action,
)

VOICE_ASSETS_DIR = 'assets/voice'


type Media = Document | Audio | Video


# https://github.com/yagop/node-telegram-bot-api/issues/544
# https://openclaw.turtleand.com/topics/telegram-voice-speed-control/
async def convert_voice(
    msg: Message,
    attachment: Media | Output,
    raw_duration: timedelta | float,
    bitrate_k: int,
    quality: bool,
) -> None:
    log.info('Attachment: %s', attachment)

    if isinstance(raw_duration, timedelta):
        duration = raw_duration.total_seconds()
    else:
        duration = raw_duration

    if isinstance(attachment, Output):
        # Path to an external file.
        file_name = os.path.basename(attachment.path)
        file_size = attachment.size
    else:
        file_name = attachment.file_name
        file_size = attachment.file_size

    attrs = []
    if file_size:
        attrs.append(f'{file_size} bytes')
    if duration > 0:
        attrs.append(f'{duration} s')
    if bitrate_k > 0:
        attrs.append(f'{bitrate_k}k')
    if quality:
        attrs.append('quality')

    attrs = ', '.join(attrs)
    log.info('Processing: %s (%s)', file_name, attrs)

    if file_name:
        info = rf'Processing: `{escape(file_name)}` \({escape(attrs)}\)'
    else:
        info = rf'Processing: _Untitled_ \({escape(attrs)}\)'

    status: Message | None = None
    settings: list[str | None] = [info]
    queue = asyncio.Queue()

    async def _refresh():
        nonlocal status

        if settings:
            text = '\n'.join(s for s in settings if s)
        else:
            text = info

        # This needs to be serialized with a queue.
        if status is None:
            status = await reply_text(msg, text, 'MarkdownV2')
        else:
            await status.edit_text(text, 'MarkdownV2')

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

        if isinstance(attachment, Output):
            log.debug('Using external file: %s', attachment)
            file_path = attachment.path
        else:
            file = await attachment.get_file()
            log.debug('File: %s', file)

            # https://github.com/aiogram/telegram-bot-api/issues/30
            file_path = file.file_path
            if file_path and os.path.isfile(file_path):
                log.debug('Using local file: %s', file_path)
            else:
                from pathvalidate import sanitize_filename

                if file_name := (file_name or file_path):
                    base, ext = os.path.splitext(file_name)
                    file_name = f'{base} [{file.file_unique_id}]{ext}'
                else:
                    file_name = file.file_unique_id

                os.makedirs(VOICE_ASSETS_DIR, exist_ok=True)
                dst = os.path.join(VOICE_ASSETS_DIR, sanitize_filename(file_name))
                src = await file.download_to_drive(custom_path=dst)
                file_path = str(src)

        log.info('Encoding voice from %s', file_path)
        result = await encode_voice(file_path, report, duration, bitrate_k, quality)

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


async def try_handle_voice(update: Update, *, parse_url: bool = False) -> bool:
    msg, arg = get_msg_arg(update)
    info = extract_media(msg) or (
        msg.reply_to_message and extract_media(msg.reply_to_message)
    )
    parsed = None
    if not info and (not parse_url or (parsed := extract_url(arg)) is None):
        return False

    with keep_chat_action(msg, ChatAction.RECORD_VOICE):
        if not info:
            # Delegate to ytdlp if the argument looks like a URL.
            assert parsed is not None
            url, arg = parsed
            output = await run_ytdlp(url, audio_only=True)
            asyncio.create_task(output.finish(msg, audio_only=True))
            info = output, output.duration

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
    if not await try_handle_voice(update, parse_url=True):
        await reply_text(
            update,
            r'Send or reply to a media message with `/voice [q]`, or use `/voice <url>`\.',
            'MarkdownV2',
        )
