import os
import math
import asyncio
from typing import Awaitable
from datetime import timedelta

from telegram import Message, Document, Audio, Video
from telegram.constants import ChatAction

from log import log
from ffmpeg import encode_voice
from ytdlp import run_ytdlp, extract_url, Output
from util import (
    create_task,
    escape,
    get_context,
    Responder,
    InlineResponder,
    EditHandle,
    VoicePayload,
    command,
)

VOICE_ASSETS_DIR = 'assets/voice'


type Media = Document | Audio | Video


# https://github.com/yagop/node-telegram-bot-api/issues/544
# https://openclaw.turtleand.com/topics/telegram-voice-speed-control/
async def convert_voice(
    rs: Responder,
    attachment: Media | Output,
    raw_duration: timedelta | float,
    bitrate_k: int,
    quality: bool,
    quiet: bool,
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

    if quiet:
        queue = None

        def report(idx: int, text: str):
            log.debug('Status %s: %s', idx, text)

    else:
        queue = asyncio.Queue()
        status: EditHandle | None = None
        settings: list[str | None] = [info]

        async def _refresh():
            nonlocal status

            if settings:
                text = '\n'.join(s for s in settings if s)
            else:
                text = info

            # This needs to be serialized with a queue.
            if status is None:
                status = await rs.reply_cached(text, 'MarkdownV2')
            else:
                await status.edit_text(text, 'MarkdownV2')

        async def worker():
            while True:
                r = await queue.get()
                if r is None:
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

        # Report initial status before downloading the file.
        queue.put_nowait(True)
        create_task(worker())

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
    duration, data, bitrate_k = await encode_voice(
        file_path, report, duration, bitrate_k, quality
    )
    if queue is not None:
        queue.put_nowait(None)
    report(2, f'Encoded into {len(data)} bytes at {bitrate_k} kbps')
    await rs.reply(
        media=VoicePayload(data, math.ceil(duration) if duration >= 0 else None)
    )


def extract_media(
    msg: Message,
) -> tuple[Media, int | timedelta] | None:
    if (media := msg.document) is not None:
        mime = media.mime_type
        log.debug('voice: document mime: %s', mime)
        if mime and (mime.startswith('audio') or mime.startswith('video')):
            return media, 0
    elif (media := msg.audio or msg.video) is not None:
        return media, media.duration


# This is sync before we must defer the `rs` before entering async, otherwise
# this would not work in `reroute_cmd`, which usually runs as a `Task` after we
# yield.
def try_handle_voice(
    msg: Message, rs: Responder, parse_url: bool = False
) -> Awaitable | None:
    arg = rs.get_arg()
    info = extract_media(msg) or (
        msg.reply_to_message and extract_media(msg.reply_to_message)
    )
    parsed = None
    if info or (parse_url and (parsed := extract_url(arg)) is not None):
        if isinstance(rs, InlineResponder):
            # XXX: A voice cannot be edited onto an existing message, so we have
            # to defer the `InlineResponder`.
            # Otherwise, the inline message would be sent before the voice is ready.
            log.info('try_handle_voice: deferring: %r', rs)

            # An inline message cannot contain two media, so we have to skip sending
            # the original audio.
            return rs.wait_until(_try_handle_voice(rs, info, parsed, arg, False))

        return _try_handle_voice(rs, info, parsed, arg)


async def _try_handle_voice(
    rs: Responder,
    info: tuple[Media | Output, float | timedelta] | None,
    parsed: tuple[str, str] | None,
    arg: str,
    finish: bool = True,
):
    with rs.keep_chat_action(ChatAction.RECORD_VOICE):
        if info is None:
            # Delegate to ytdlp if the argument looks like a URL.
            assert parsed is not None
            url, arg = parsed
            output = await run_ytdlp(url, audio_only=True)
            if finish:
                create_task(output.finish(rs, audio_only=True))
            info = output, output.duration

        if get_context().sender_is_host():
            quality = 'Q' not in arg
        elif arg:
            quality = 'q' in arg
        else:
            quality = False

        quiet = 's' in arg

        if (p := arg.find('k')) > 0 and arg[:p].isdigit():
            bitrate_k = int(arg[:p])
        else:
            bitrate_k = 0

        await convert_voice(rs, *info, bitrate_k, quality, quiet)


@command(public=True)
def handle_voice(msg: Message, rs: Responder):
    fut = try_handle_voice(msg, rs, parse_url=True)
    if fut is not None:
        return fut
    return rs.reply_cached(
        r'Send or reply to a media message with `/voice [q]`, or use `/voice <url>`\.',
        'MarkdownV2',
    )
