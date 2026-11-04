import asyncio
import math
import os
import re

from dataclasses import dataclass
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
from .ytdlp import Output, extract_url, is_video, run_ytdlp

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
    try:
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
        report(4, f'Encoded into {len(r.data)} bytes at {r.bitrate_k} kbps in {r.iterations} iters')
    finally:
        if queue is not None:
            queue.put_nowait(None)
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
NIGHTCORE_RE = re.compile(r'(?:nc|nightcore)=?(\d+(?:\.\d+)?)?')
FLAGS_RE = re.compile(r'(?:(\d+)k)?([Qqsd]*)')


def parse_time(s: str) -> float:
    parts = s.split(':')
    match len(parts):
        case 1:
            return float(parts[0])
        case 2:
            return int(parts[0]) * 60 + float(parts[1])
        case _:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


@dataclass(frozen=True, slots=True)
class VoiceArgs:
    speed: float = 1
    start_time: float | None = None
    end_time: float | None = None
    bitrate_k: int = 0
    flags: str = ''


def parse_voice_args(arg: str) -> VoiceArgs | None:
    speed = 1
    start_time = end_time = None
    bitrate_k = 0
    flags = ''
    for word in arg.split():
        if m := NIGHTCORE_RE.fullmatch(word):
            speed = float(m[1]) if m[1] else 1.5
        elif word == 'drum':
            flags += 'd'
        elif (m := CROP_RE.fullmatch(word)) and (m[1] or m[2]):
            start_time = parse_time(m[1]) if m[1] else None
            end_time = parse_time(m[2]) if m[2] else None
        elif m := FLAGS_RE.fullmatch(word):
            if m[1]:
                bitrate_k = int(m[1])
            flags += m[2]
        else:
            return None
    return VoiceArgs(speed, start_time, end_time, bitrate_k, flags)


type Source = tuple[Media | Output, float | timedelta]


def voice_request(
    rs: Responder, arg: str, as_command: bool
) -> tuple[Source | None, str | None, VoiceArgs] | None:
    '''The media or URL that the message of rs with arg converts, and the parameters arg holds.
    Without a command, a URL converts only with parameters other than `q` alone, since a URL
    alone or with `q` downloads the video.'''
    url = None
    info = rs.extract(extract_media)
    if info is None:
        if (parsed := extract_url(arg)) is None or not as_command and is_video(parsed[1]):
            return None
        url, arg = parsed
    if (args := parse_voice_args(arg)) is None:
        return None
    return info, url, args


async def _handle_voice(
    rs: Responder,
    info: Source | None,
    url: str | None,
    args: VoiceArgs,
    finish: bool = True,
):
    with rs.keep_chat_action(ChatAction.RECORD_VOICE):
        if info is None:
            assert url is not None
            output = await run_ytdlp(url, audio_only=True)
            if finish:
                create_task(output.finish(rs, audio_only=True))
            info = output, output.duration

        host = get_context().sender_is_host()
        quality = 'Q' not in args.flags if host else 'q' in args.flags

        await convert_voice(
            rs,
            *info,
            args.bitrate_k,
            quality,
            's' in args.flags,
            args.start_time,
            args.end_time,
            speed=args.speed,
            drum='d' in args.flags,
        )


async def _reply_usage(rs: Responder):
    await rs.reply_cached(
        r'Send or reply to a media message with `/voice [q]`, or use `/voice <url>`\.',
        'MarkdownV2',
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
    msg = rs.get_message()
    req = voice_request(rs, arg, as_command)
    if req is None and not as_command and msg is not None and (own := extract_media(msg)):
        # A forwarded media keeps its author's caption, which is rarely a parameter.
        req = own, None, VoiceArgs()
    if req is None:
        if as_command:
            await _reply_usage(rs)
        # A reply to media whose text is not parameters chats about the media.
        return rs.extract(extract_media) is not None

    # An inline message cannot contain two media, so we have to skip sending
    # the original audio when deferred.
    await _handle_voice(rs, *req, flush is None)
    return True
