import os
import math
import asyncio
import logging
from io import BytesIO
from typing import Any, cast
from urllib.parse import urlparse

from telegram import Message, Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction

from util import log, get_msg_arg, reply_text, keep_chat_action


def truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 3)] + '...'


def is_http_url(text: str) -> bool:
    p = urlparse(text)
    return p.scheme in ('http', 'https') and bool(p.netloc)


MAX_FILE_SIZE = 50 << 20
ASSETS_DIR = 'assets/yt'
OUTTMPL = os.path.join(ASSETS_DIR, '%(title).120B [%(id)s].%(ext)s')

# The thumbnail should be in JPEG format and less than 200 kB in size.
# A thumbnail's width and height should not exceed 320.
THUMB_MAX_BYTES = 200 << 10
THUMB_MAX_SIDE = 320
THUMB_QUALITY_STEPS = (90, 80, 70, 60, 50, 40, 30)
THUMB_SCALE_STEPS = (1.0, 0.85, 0.7, 0.55, 0.4)


def _prepare_thumbnail(data: bytes) -> bytes | None:
    from PIL import Image

    try:
        with Image.open(BytesIO(data)) as src:
            base = src.convert('RGB')
    except Exception as e:
        log.exception('Failed to decode thumbnail image: %s', e)
        return None

    resample = Image.Resampling.LANCZOS
    base.thumbnail((THUMB_MAX_SIDE, THUMB_MAX_SIDE), resample=resample)

    for scale in THUMB_SCALE_STEPS:
        img = base.copy()
        if scale < 1:
            new_size = (
                max(1, int(img.width * scale)),
                max(1, int(img.height * scale)),
            )
            img = img.resize(new_size, resample=resample)

        for quality in THUMB_QUALITY_STEPS:
            out = BytesIO()
            img.save(out, format='JPEG', optimize=True, quality=quality)

            if len(candidate := out.getvalue()) <= THUMB_MAX_BYTES:
                log.info(
                    'Thumbnailed into %d bytes (s=%.2f, q=%d)',
                    len(candidate),
                    scale,
                    quality,
                )
                return candidate

    log.warning('Thumbnail too large: %d bytes', len(data))
    return data


def _get_download_path(info: dict[str, Any]) -> str:
    for item in info.get('requested_downloads') or ():
        if path := item.get('filepath'):
            return path
    if path := info.get('_filename'):
        return path
    raise FileNotFoundError('yt-dlp finished but no output file was found')


def _get_thumbnail_path(info: dict[str, Any]) -> str | None:
    for thumb in reversed(info.get('thumbnails') or ()):
        log.debug('try thumb: %s', thumb)
        if path := thumb.get('filepath'):
            if os.path.isfile(path):
                return path


class Output:
    def __init__(self, info: dict[str, Any]):
        self.info = info
        self.path = path = _get_download_path(info)

        self.size = size = os.path.getsize(path)
        if size > MAX_FILE_SIZE:
            raise ValueError(f'File too large for Telegram ({size})')

        self.duration = float(info.get('duration', 0))
        self.thumbnail_path = _get_thumbnail_path(info)
        log.info('ytdlp: %s / %s / thumb=%s', path, self.duration, self.thumbnail_path)

    def __str__(self) -> str:
        return f'<ytdlp.Output: {self.path}, {self.duration}, {self.thumbnail_path}>'

    __repr__ = __str__

    async def finish(self, msg: Message, *, audio_only: bool = False):
        info = self.info
        path = self.path

        def get(key: str) -> str | None:
            if (val := info.get(key)) is not None:
                return str(val).strip()

        title = get('title') or get('id')
        performer = get('uploader') or get('channel') or get('creator')

        if title:
            ext = os.path.splitext(path)[1].lower()
            if performer:
                name = f'{title} - {performer}{ext}'
            else:
                name = f'{title}{ext}'
        else:
            name = os.path.basename(path)

        name = truncate(name, 64)
        duration = math.ceil(self.duration) if self.duration > 0 else None

        thumbnail = raw_thumbnail = None
        if self.thumbnail_path:
            try:
                with open(self.thumbnail_path, 'rb') as fp:
                    raw_thumbnail = fp.read()
            except Exception as e:
                log.exception('Failed to read thumbnail %s: %s', self.thumbnail_path, e)
            else:
                thumbnail = _prepare_thumbnail(raw_thumbnail)

        log.info(
            'finish: %s %s thumb=%s/%s',
            name,
            duration,
            raw_thumbnail and len(raw_thumbnail),
            thumbnail and len(thumbnail),
        )
        with open(path, 'rb') as fp:
            if audio_only:
                performer = truncate(performer, 64) if performer else None
                await msg.reply_audio(
                    fp,
                    filename=name,
                    duration=duration,
                    thumbnail=thumbnail,
                    title=title,
                    performer=performer,
                    do_quote=True,
                    allow_sending_without_reply=True,
                )
            else:
                await msg.reply_video(
                    fp,
                    filename=name,
                    duration=duration,
                    thumbnail=thumbnail,
                    cover=raw_thumbnail,
                    supports_streaming=True,
                    do_quote=True,
                    allow_sending_without_reply=True,
                )


def ytdlp_task(url: str, audio_only: bool) -> Output:
    import yt_dlp

    os.makedirs(ASSETS_DIR, exist_ok=True)

    fmt = 'bestaudio/best' if audio_only else 'bv*+ba/b'
    debug = log.isEnabledFor(logging.DEBUG)

    opts = {
        'quiet': not debug,
        'no_warnings': not debug,
        'verbose': debug,
        'noplaylist': True,
        'restrictfilenames': False,
        'format': fmt,
        'outtmpl': OUTTMPL,
        'writethumbnail': True,
        'max_filesize': MAX_FILE_SIZE,
    }

    with yt_dlp.YoutubeDL(cast(Any, opts)) as ydl:
        info = ydl.extract_info(url)

    if not info:
        raise RuntimeError('yt-dlp returned no info')

    return Output(cast(dict[str, Any], info))


async def run_ytdlp(url: str, *, audio_only: bool = False) -> Output:
    return await asyncio.to_thread(ytdlp_task, url, audio_only)


async def _handle_yt(update: Update, *, audio_only: bool = False):
    msg, arg = get_msg_arg(update)
    if not arg:
        usage = '/yta' if audio_only else '/yt'
        await reply_text(msg, f'Usage: {usage} <url>')
        return

    url = arg.split()[0]
    if not is_http_url(url):
        await reply_text(msg, 'Please provide a valid URL.')
        return

    action = ChatAction.RECORD_VOICE if audio_only else ChatAction.RECORD_VIDEO
    with keep_chat_action(msg, action):
        try:
            output = await run_ytdlp(url, audio_only=audio_only)
            await output.finish(msg, audio_only=audio_only)
        except Exception as e:
            log.exception('ytdlp failed for %s', url)
            error_msg = truncate(f'Download failed: {type(e).__name__}: {e}', 500)
            await reply_text(msg, error_msg)


async def handle_yt(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await _handle_yt(update, audio_only=False)


async def handle_yta(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await _handle_yt(update, audio_only=True)
