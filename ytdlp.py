import os
import re
import math
import json
import asyncio
import functools
from io import BytesIO
from typing import Any, cast, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

from telegram import (
    Bot,
    Video,
    Audio,
    Document,
    Update,
    Message,
    InlineQueryResultCachedAudio,
    InlineQueryResultCachedVideo,
    InlineQueryResultCachedDocument,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQuery,
    InlineQueryResultArticle,
    InlineQueryResultCachedVoice,
    InputMediaAudio,
    InputMediaVideo,
    InputMediaDocument,
    InputTextMessageContent,
    SwitchInlineQueryChosenChat,
)
from telegram.ext import ContextTypes
from telegram.constants import ChatAction

from util import log, is_debug, get_msg_arg, reply_text, keep_chat_action
from render_context import LRUDict
from ffmpeg import run_ffmpeg

if TYPE_CHECKING:
    from yt_dlp import YoutubeDL

REG_URL = re.compile(
    r'(?:(?:https?|voice)://|(www\.))([-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*))'
)

CANONICAL_KEEP_QUERY = frozenset(('v', 'list', 't', 'index', 'p'))


def canonicalize_url(url: str) -> str:
    from urllib.parse import urlparse, urlencode, parse_qsl, urlunparse

    u = urlparse(url)
    qs = [
        (k, v)
        for k, v in parse_qsl(u.query)
        if k in CANONICAL_KEEP_QUERY and (k, v) != ('p', '1')
    ]
    qs.sort()
    return urlunparse(u._replace(query=urlencode(qs), fragment='', scheme='https'))


def matched_url(m: re.Match) -> str:
    url = ''.join(s for s in m.groups() if s is not None)
    return canonicalize_url('https://' + url.rstrip('#/?'))


def truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 3)] + '...'


def extract_url(text: str) -> tuple[str, str] | None:
    url = None

    def repl(m: re.Match) -> str:
        nonlocal url
        url = matched_url(m)
        if m.string[m.start()] == 'v':
            return 'voice'
        return ''

    left = REG_URL.sub(repl, text.strip(), count=1)
    if url is not None:
        return url, left.strip()


def find_url(text: str) -> str | None:
    m = REG_URL.search(text.strip())
    return m and matched_url(m)


MAX_FILE_SIZE = 50 << 20
ASSETS_DIR = 'assets/yt'
OUTTMPL = os.path.join(ASSETS_DIR, '%(title).120B [%(id)s].%(ext)s')

# The thumbnail should be in JPEG format and less than 200 kB in size.
# A thumbnail's width and height should not exceed 320.
THUMB_MAX_BYTES = 200 << 10
THUMB_MAX_SIDE = 320
THUMB_QUALITY_STEPS = (90, 80, 70, 60, 50, 40, 30)
THUMB_SCALE_STEPS = (1.0, 0.85, 0.7, 0.55, 0.4)

YT_STAGING_CHAT_ID = int(os.environ['YT_STAGING_CHAT_ID'])
YT_STAGING_MESSAGE_THREAD_ID = (
    int(os.environ.get('YT_STAGING_MESSAGE_THREAD_ID', 0)) or None
)

# max_workers=1 to serialize yt-dlp invocations.
_executor = ThreadPoolExecutor(max_workers=1)
_instances: list['YoutubeDL | None'] = [None, None]

# For passing inline query results.
_media_cache: LRUDict[str, Video | Audio | Document] = LRUDict()
_voice_cache: LRUDict[str, tuple[str, str]] = LRUDict()


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


VIDEO_NOTE_MAX_DURATION = 60
VIDEO_NOTE_SIDE = 640

# Telegram rejects video notes larger than 12 MiB.
VIDEO_NOTE_MAX_BYTES = 12 << 20
VIDEO_NOTE_AUDIO_BITRATE_K = 64


async def _convert_video_note(src: str, duration: float) -> str:
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


def media_duration(duration: float) -> int | None:
    return math.ceil(duration) if duration > 0 else None


class Output:
    def __init__(self, info: dict[str, Any]):
        self.info = info
        self.path = path = _get_download_path(info)

        self.size = size = os.path.getsize(path)
        if size > MAX_FILE_SIZE:
            raise ValueError(f'File too large for Telegram ({size})')

        self.duration = float(info.get('duration', 0))
        self.thumbnail_path = _get_thumbnail_path(info)
        log.info(
            'ytdlp: %s / %s bytes / %s secs / %s',
            path,
            size,
            self.duration,
            self.thumbnail_path,
        )

    def __str__(self) -> str:
        return f'<ytdlp.Output: {self.path}, {self.duration}, {self.thumbnail_path}>'

    __repr__ = __str__

    def _get(self, key: str) -> str | None:
        if (val := self.info.get(key)) is not None:
            return str(val).strip()

    @property
    def title(self) -> str | None:
        return self._get('title') or self._get('id')

    @property
    def performer(self) -> str | None:
        return self._get('uploader') or self._get('channel') or self._get('creator')

    @property
    def url(self) -> str | None:
        url = self.info.get('webpage_url')
        return url and canonicalize_url(url)

    def _thumbnail(self) -> tuple[bytes | None, bytes | None]:
        if not self.thumbnail_path:
            return None, None

        raw, thumb = None, None
        try:
            with open(self.thumbnail_path, 'rb') as fp:
                raw = fp.read()
            thumb = _prepare_thumbnail(raw)
        except Exception as e:
            log.exception('Failed to thumbnail: %s: %s', self.thumbnail_path, e)

        return raw, thumb

    def get_name(self) -> str:
        title = self.title
        performer = self.performer
        if title:
            if performer:
                return f'{title} - {performer}'
            return title
        return os.path.basename(self.path)

    async def finish(
        self, msg_or_bot: Message | Bot, *, audio_only: bool = False
    ) -> Message:
        path = self.path
        title = self.title
        performer = self.performer

        if title:
            ext = os.path.splitext(path)[1].lower()
            if performer:
                name = f'{title} - {performer}{ext}'
            else:
                name = f'{title}{ext}'
        else:
            name = os.path.basename(path)

        name = truncate(name, 64)
        duration = media_duration(self.duration)

        raw_thumbnail, thumbnail = self._thumbnail()

        log.info(
            'finish: %s %s thumb=%s/%s',
            name,
            duration,
            raw_thumbnail and len(raw_thumbnail),
            thumbnail and len(thumbnail),
        )

        if isinstance(msg_or_bot, Message):
            wrap = lambda f: functools.partial(
                f, do_quote=True, allow_sending_without_reply=True
            )
            send_audio = msg_or_bot.reply_audio
            send_video = msg_or_bot.reply_video
        else:
            wrap = lambda f: functools.partial(
                f,
                YT_STAGING_CHAT_ID,
                message_thread_id=YT_STAGING_MESSAGE_THREAD_ID,
                disable_notification=True,
            )
            send_audio = msg_or_bot.send_audio
            send_video = msg_or_bot.send_video

        with open(self.path, 'rb') as fp:
            if audio_only:
                performer = truncate(performer, 64) if performer else None
                return await wrap(send_audio)(
                    fp,
                    filename=name,
                    duration=duration,
                    thumbnail=thumbnail,
                    title=title,
                    performer=performer,
                )
            else:
                return await wrap(send_video)(
                    fp,
                    filename=name,
                    duration=duration,
                    thumbnail=thumbnail,
                    cover=raw_thumbnail,
                    supports_streaming=True,
                )

    async def finish_video_note(self, msg: Message) -> Message:
        dst = await _convert_video_note(self.path, self.duration)

        duration = media_duration(self.duration)
        if duration is not None:
            duration = min(duration, VIDEO_NOTE_MAX_DURATION)

        _, thumbnail = self._thumbnail()

        with open(dst, 'rb') as fp:
            return await msg.reply_video_note(
                fp,
                duration=duration,
                length=VIDEO_NOTE_SIDE,
                thumbnail=thumbnail,
                do_quote=True,
                allow_sending_without_reply=True,
            )


def get_ytdlp(audio_only: bool) -> 'YoutubeDL':
    from yt_dlp import YoutubeDL

    if (ydl := _instances[audio_only]) is not None:
        return ydl

    os.makedirs(ASSETS_DIR, exist_ok=True)

    # Prefer Telegram-streamable containers.
    if audio_only:
        fmt = 'bestaudio[ext=m4a]/bestaudio/best'
        postprocessors = ({'key': 'FFmpegExtractAudio', 'preferredcodec': 'm4a'},)
    else:
        fmt = 'bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b'
        postprocessors = ({'key': 'FFmpegVideoConvertor', 'preferedformat': 'mp4'},)

    # Do not pass 'logger' here, or progress bars will break.
    opts = {
        'quiet': not is_debug,
        'noprogress': not is_debug,
        'verbose': is_debug,
        'noplaylist': True,
        'keepvideo': True,
        'restrictfilenames': False,
        'format': fmt,
        'postprocessors': postprocessors,
        'outtmpl': OUTTMPL,
        'writethumbnail': True,
        'max_filesize': MAX_FILE_SIZE,
    }

    _instances[audio_only] = ydl = YoutubeDL(cast(Any, opts))
    return ydl


# Serialized to avoid conflicts inside ASSETS_DIR and races on _instances,
# since _executor has max_workers=1.
def ytdlp_task(url: str, audio_only: bool) -> Output:
    log.info('yt-dlp: invoking %s (audio_only=%s)', url, audio_only)
    info = get_ytdlp(audio_only).extract_info(url)

    if is_debug and info:
        default = lambda o: f'<default: {type(o).__name__}: {o!r}>'
        with open('last_ytdlp_info.json', 'w', encoding='utf-8') as fp:
            json.dump(info, fp, ensure_ascii=False, indent=2, default=default)

    if not info:
        raise RuntimeError('yt-dlp returned no info')

    return Output(cast(dict[str, Any], info))


async def run_ytdlp(url: str, *, audio_only: bool = False) -> Output:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, ytdlp_task, url, audio_only)


async def _handle_yt(
    update: Update,
    cmd: str,
    action: ChatAction,
    *,
    audio_only: bool = False,
    video_note: bool = False,
):
    msg, arg = get_msg_arg(update)
    if not arg:
        await reply_text(msg, f'Usage: {cmd} <url>')
        return

    if (url := find_url(arg)) is None:
        await reply_text(msg, 'Please provide a valid URL.')
        return

    with keep_chat_action(msg, action):
        try:
            output = await run_ytdlp(url, audio_only=audio_only)
            if video_note:
                await output.finish_video_note(msg)
            else:
                await output.finish(msg, audio_only=audio_only)
        except Exception as e:
            log.exception('ytdlp failed for %s', url)
            error_msg = truncate(f'Download failed: {type(e).__name__}: {e}', 500)
            await reply_text(msg, error_msg)


async def handle_yt(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await _handle_yt(update, '/yt', ChatAction.RECORD_VIDEO)


async def handle_yta(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await _handle_yt(update, '/yta', ChatAction.RECORD_VOICE, audio_only=True)


async def handle_ytn(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await _handle_yt(update, '/ytn', ChatAction.RECORD_VIDEO_NOTE, video_note=True)


def make_markup(text: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup.from_button(
        InlineKeyboardButton(text, callback_data='noop')
    )


async def handle_yt_inline_query(query: InlineQuery, parsed: tuple[str, str]):
    url, arg = parsed
    log.info('handle_yt_inline_query: %s / %s', url, arg)
    caption = None if 'q' in arg else url

    if 'voice' in arg and (voice := _voice_cache.get(url)):
        # `voice://` is set from `_finish_voice`.
        file_id, title = voice
        result = InlineQueryResultCachedVoice(
            id='noop', title=title, voice_file_id=file_id, caption=caption
        )
        await query.answer((result,))
        return

    media = _media_cache.get(url)
    results = []
    cached_time = None

    if isinstance(media, Video):
        results.append(
            InlineQueryResultCachedVideo(
                id='noop_1',
                video_file_id=media.file_id,
                title='📹 Video: ' + (media.file_name or url),
                caption=caption,
            )
        )
    elif isinstance(media, Document):
        results.append(
            InlineQueryResultCachedDocument(
                id='noop_2',
                document_file_id=media.file_id,
                title='📄 File: ' + (media.file_name or url),
                caption=caption,
            )
        )
    else:
        cached_time = 0
        results.append(
            InlineQueryResultArticle(
                id='yt_video',
                title='📹 Video',
                input_message_content=InputTextMessageContent(url),
                reply_markup=make_markup('📹 Downloading video...'),
            )
        )

    if isinstance(media, Audio):
        results.append(
            InlineQueryResultCachedAudio(
                id='noop_3',
                audio_file_id=media.file_id,
                caption=caption,
            )
        )
    else:
        cached_time = 0
        results.append(
            InlineQueryResultArticle(
                id='yt_audio',
                title='🎵 Audio',
                input_message_content=InputTextMessageContent(url),
                reply_markup=make_markup('🎵 Downloading audio...'),
            )
        )

    if voice := _voice_cache.get(url):
        file_id, title = voice
        results.append(
            InlineQueryResultCachedVoice(
                id='noop_4',
                title='🎤 Voice:' + title,
                voice_file_id=file_id,
                caption=caption,
            )
        )
    else:
        cached_time = 0
        results.append(
            InlineQueryResultArticle(
                id='yt_voice',
                title='🎤 Voice',
                input_message_content=InputTextMessageContent(url),
                reply_markup=make_markup('🎤 Encoding voice...'),
            )
        )

    await query.answer(results, cache_time=cached_time)


async def _finish_voice(
    bot: Bot,
    output: Output,
    result: tuple[float, bytes, int],
    url: str,
) -> InlineKeyboardMarkup | None:
    duration, data, bitrate = result
    duration = media_duration(duration)

    msg = await bot.send_voice(
        YT_STAGING_CHAT_ID,
        data,
        message_thread_id=YT_STAGING_MESSAGE_THREAD_ID,
        duration=duration,
        disable_notification=True,
    )

    if voice := msg.voice:
        _voice_cache[url] = (voice.file_id, output.get_name())
        log.info('Cached voice for %s: %s', url, voice)
    else:
        log.error('Staging returned no voice: %s', msg)
        return None

    # Bypass cache for the previous inline query result. Also detected above in
    # `handle_yt_inline_query()`.
    # `matched_url()` and `canonicalize_url()` ensures the url starts with 'https://'.
    query = 'voice://' + url[8:]

    row = (
        InlineKeyboardButton(
            f'✅ Send here ({bitrate} kbps)', switch_inline_query_current_chat=query
        ),
        InlineKeyboardButton(
            '🎤 Send to ...',
            switch_inline_query_chosen_chat=SwitchInlineQueryChosenChat(
                query,
                allow_user_chats=True,
                allow_bot_chats=True,
                allow_group_chats=True,
                allow_channel_chats=True,
            ),
        ),
    )
    return InlineKeyboardMarkup.from_row(row)


async def handle_yt_chosen_result(
    bot: Bot,
    result_id: str,
    parsed: tuple[str, str],
    inline_message_id: str,
):
    url, arg = parsed
    log.info('handle_yt_chosen_result: %s: %s / %s', result_id, url, arg)
    markup = None

    try:
        is_voice = result_id == 'yt_voice'
        audio_only = is_voice or result_id == 'yt_audio'
        output = await run_ytdlp(url, audio_only=audio_only)
        url = output.url or url
        caption = None if 'q' in arg else url

        if is_voice:
            from voice import encode_voice

            async def worker():
                nonlocal markup
                result = await encode_voice(
                    output.path, lambda *_: None, output.duration
                )
                markup = await _finish_voice(bot, output, result, url)
                await bot.edit_message_caption(
                    inline_message_id=inline_message_id,
                    caption=caption,
                    reply_markup=markup,
                )

            asyncio.create_task(worker())

    except Exception as e:
        log.exception('handle_yt_chosen_result: failed for %s', url)
        error_msg = truncate(f'❌ {type(e).__name__}: {e}', 200)
        await bot.edit_message_caption(
            caption=error_msg, inline_message_id=inline_message_id
        )
        return

    # Upload to staging chat to get file_id, then edit inline message.
    staging = await output.finish(bot, audio_only=audio_only)

    if media := staging.audio:
        _media_cache[url] = media
        input_media = InputMediaAudio(media=media, caption=caption)
    elif media := staging.video:
        _media_cache[url] = media
        input_media = InputMediaVideo(
            media=media, caption=caption, supports_streaming=True
        )
    elif media := staging.document:
        _media_cache[url] = media
        input_media = InputMediaDocument(media=media, caption=caption)
    else:
        log.error('Staging returned no media: %s', staging)
        input_media = None

    if input_media:
        log.info('Media ready: %s', media)
        await bot.edit_message_media(
            input_media, inline_message_id=inline_message_id, reply_markup=markup
        )
    else:
        await bot.edit_message_caption(
            inline_message_id=inline_message_id,
            caption='Unknown error for ' + url,
            reply_markup=markup,
        )
