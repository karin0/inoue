import asyncio
import contextlib
import math
import os
import re

from datetime import timedelta

from telegram import Audio, Document, Message, Video, Voice
from telegram.constants import ChatAction

from bot import (
    EditHandle,
    MessageArg,
    RequireDefer,
    Responder,
    VoicePayload,
    command,
    create_task,
    escape,
)

from .ctx import get_context
from .ffmpeg import encode_voice
from .log import log
from .ytdlp import Output, extract_url, run_ytdlp

VOICE_ASSETS_DIR = 'assets/voice'

type Media = Document | Audio | Video | Voice


# https://github.com/yagop/node-telegram-bot-api/issues/544
# https://openclaw.turtleand.com/topics/telegram-voice-speed-control/
async def convert_voice(
    rs: Responder,
    attachment: Media | Output,
    raw_duration: timedelta | float,
    bitrate_k: int,
    quality: bool,
    quiet: bool,
    start_time: float | None = None,
    end_time: float | None = None,
    speed: float = 1,
    drum: bool = False,
) -> None:
    log.info('Attachment: %s', attachment)

    duration = raw_duration.total_seconds() if isinstance(raw_duration, timedelta) else raw_duration

    if isinstance(attachment, Output):
        # Path to an external file.
        file_name = os.path.basename(attachment.path)
        file_size = attachment.size
    else:
        file_name = attachment.file_name if not isinstance(attachment, Voice) else None
        file_size = attachment.file_size

    attrs = []
    if file_size:
        attrs.append(f'{file_size} bytes')
    if duration > 0:
        attrs.append(f'{duration} s')
    if start_time is not None or end_time is not None:
        attrs.append(f'crop:{start_time or 0}-{end_time or ""}')
    if bitrate_k > 0:
        attrs.append(f'{bitrate_k}k')
    if quality:
        attrs.append('quality')
    if drum:
        attrs.append('drum')
    if speed != 1:
        attrs.append(f'speed={speed}')

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

            text = '\n'.join(s for s in settings if s) if settings else info

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
    r = await encode_voice(
        file_path,
        report,
        duration,
        bitrate_k,
        quality,
        start_time,
        end_time,
        speed=speed,
        drum=drum,
    )
    if queue is not None:
        queue.put_nowait(None)
    report(2, f'Encoded into {len(r.data)} bytes at {r.bitrate_k} kbps in {r.iterations} iters')
    await rs.reply(media=VoicePayload(r.data, math.ceil(r.duration) if r.duration >= 0 else None))


def extract_media(msg: Message) -> tuple[Media, int | timedelta] | None:
    if (media := msg.document) is not None:
        mime = media.mime_type
        log.debug('voice: document mime: %s', mime)
        if mime and (mime.startswith(('audio', 'video'))):
            return media, 0
    elif (media := msg.audio or msg.video or msg.voice) is not None:
        return media, media.duration


CROP_RE = re.compile(r'((?:\d+:){0,2}\d+(?:\.\d+)?)?-((?:\d+:){0,2}\d+(?:\.\d+)?)?')


def parse_time(s: str) -> float:
    parts = s.split(':')
    match len(parts):
        case 1:
            return float(parts[0])
        case 2:
            return int(parts[0]) * 60 + float(parts[1])
        case _:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def extract_crop(arg: str) -> tuple[str, float | None, float | None] | None:
    new_args = []
    interval = None
    for word in arg.split():
        if interval is None and '-' in word and (m := CROP_RE.fullmatch(word)) is not None:
            start = m.group(1)
            end = m.group(2)
            if start or end:
                interval = (parse_time(start) if start else None, parse_time(end) if end else None)
                continue
        new_args.append(word)
    if interval is not None:
        return ' '.join(new_args), *interval


def extract_effects(arg: str) -> tuple[str, float, bool]:
    speed = 1
    drum = False

    if arg:
        if m := re.search(r'\b(?:nc|nightcore)(?:=)?([0-9.]+)?\b', arg):
            speed = 1.5
            if speed_str := m.group(1):
                with contextlib.suppress(ValueError):
                    speed = float(speed_str)
            # Remove the matched nightcore part from arg
            arg = (arg[: m.start()].rstrip() + ' ' + arg[m.end() :].lstrip()).strip()

        drum = 'd' in arg

    return arg, speed, drum


async def _handle_voice(
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

        # Parse voice effects parameters
        arg, speed, drum = extract_effects(arg)

        # Parse crop interval from arg
        if arg and (r := extract_crop(arg)) is not None:
            arg, start_time, end_time = r
        else:
            start_time = end_time = None

        if get_context().sender_is_host():
            quality = 'Q' not in arg
        elif arg:
            quality = 'q' in arg
        else:
            quality = False

        quiet = 's' in arg

        bitrate_k = int(arg[:p]) if (p := arg.find('k')) > 0 and arg[:p].isdigit() else 0

        await convert_voice(
            rs,
            *info,
            bitrate_k,
            quality,
            quiet,
            start_time,
            end_time,
            speed=speed,
            drum=drum,
        )


# XXX: No `InputMediaVoice` exists, so the voice result cannot be edited onto
# an inline message.
# When `rs` is an `InlineResponder`, the caller must defer the emission until
# the voice is ready.
# Otherwise, the inline message would be sent before the voice is ready.
@command(public=True)
async def handle_voice(
    rs: Responder, arg: MessageArg, flush: RequireDefer, as_command: bool = True
) -> bool:
    parsed = None
    if (info := rs.extract(extract_media)) is not None or (
        as_command and (parsed := extract_url(arg)) is not None
    ):
        # An inline message cannot contain two media, so we have to skip sending
        # the original audio when deferred.
        await _handle_voice(rs, info, parsed, arg, flush is None)
        return True

    if as_command:
        await rs.reply_cached(
            r'Send or reply to a media message with `/voice [q]`, or use `/voice <url>`\.',
            'MarkdownV2',
        )
    return False
