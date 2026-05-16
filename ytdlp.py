import os
import re
import math
import json
import string
import asyncio
from io import BytesIO
from pathlib import Path
from typing import Any, Awaitable, cast, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

from telegram import (
    Video,
    Audio,
    Document,
    Message,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQuery,
    InlineQueryResultArticle,
    InputTextMessageContent,
    SwitchInlineQueryChosenChat,
)
from telegram.constants import ChatAction

from dispatch import MessageArg, command
from util import (
    log,
    is_debug,
    bot,
    create_task,
    get_context,
    MediaPayload,
    CachedPayload,
    AudioPayload,
    VideoPayload,
    VoicePayload,
    DocumentPayload,
    Responder,
    EditHandle,
)
from render_context import LRUDict
from ffmpeg import (
    encode_voice,
    encode_video_note,
    VIDEO_NOTE_MAX_DURATION,
    VIDEO_NOTE_SIDE,
)

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
    return urlunparse(
        u._replace(query=urlencode(qs), fragment='', scheme='https')
    ).rstrip('#/?')


def matched_url(m: re.Match) -> str:
    url = ''.join(s for s in m.groups() if s is not None)
    return canonicalize_url('https://' + url)


def truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 3)] + '...'


BASE58_CHARS = frozenset(string.ascii_letters + string.digits) - frozenset('0OIl')
YT_CHARS = frozenset(string.ascii_letters + string.digits + '-_')


def restore_url(s: str) -> str | None:
    if len(s) > 2 and s.startswith('av') and s[2] != '0' and s[2:].isdigit():
        return 'https://www.bilibili.com/video/' + s

    if len(s) == 12 and s.startswith('BV1'):
        if all(c in BASE58_CHARS for c in s[3:]):
            return 'https://www.bilibili.com/video/' + s

    if len(s := s.rstrip('=')) == 11 and all(c in YT_CHARS for c in s):
        return 'https://www.youtube.com/watch?v=' + s


def extract_url(text: str) -> tuple[str, str] | None:
    if not (parts := text.split(None, 1)):
        return None

    if (url := restore_url(parts[0])) is not None:
        return url, parts[1] if len(parts) > 1 else ''

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
    if (r := extract_url(text)) is not None:
        return r[0]


MAX_FILE_SIZE = 50 << 20
ASSETS_DIR = 'assets/yt'
OUTTMPL = os.path.join(ASSETS_DIR, '%(title).120B [%(id)s].%(ext)s')

# The thumbnail should be in JPEG format and less than 200 kB in size.
# A thumbnail's width and height should not exceed 320.
THUMB_MAX_BYTES = 200 << 10
THUMB_MAX_SIDE = 320
THUMB_QUALITY_STEPS = (90, 80, 70, 60, 50, 40, 30)
THUMB_SCALE_STEPS = (1.0, 0.85, 0.7, 0.55, 0.4)

# max_workers=1 to serialize yt-dlp invocations.
_executor = ThreadPoolExecutor(max_workers=1)
_instances: list['YoutubeDL | None'] = [None, None]

# For passing inline query results.
_media_cache: LRUDict[str, CachedPayload] = LRUDict()
_voice_cache: LRUDict[str, tuple[CachedPayload, str]] = LRUDict()

type Media = Video | Audio | Document
MEDIA_TYPES = (Video, Audio, Document)


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


def media_duration(duration: float) -> int | None:
    return math.ceil(duration) if duration > 0 else None


class Output:
    __slots__ = ('info', 'path', 'size', 'duration', 'thumbnail_path')

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

    def _payload(self, audio_only: bool) -> MediaPayload:
        path = Path(self.path)
        title = self.title
        performer = self.performer

        if title:
            ext = path.suffix.lower()
            if performer:
                name = f'{title} - {performer}{ext}'
            else:
                name = f'{title}{ext}'
        else:
            name = path.name

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

        if audio_only:
            return AudioPayload(
                path,
                duration=duration,
                filename=name,
                title=title,
                performer=truncate(performer, 64) if performer else None,
                thumbnail=thumbnail,
            )
        return VideoPayload(
            path,
            duration=duration,
            filename=name,
            thumbnail=thumbnail,
            cover=raw_thumbnail,
        )

    def finish(
        self,
        rs: Responder,
        *,
        audio_only: bool = False,
        caption: str | None = None,
    ) -> Awaitable[EditHandle | None]:
        return rs.reply(caption, media=self._payload(audio_only))

    async def finish_video_note(self, msg: Message) -> Message:
        dst = await encode_video_note(self.path, self.duration)

        duration = media_duration(self.duration)
        if duration is not None:
            duration = min(duration, VIDEO_NOTE_MAX_DURATION)

        _, thumbnail = self._thumbnail()

        with open(dst, 'rb') as fp:
            # XXX: This is not implemented as a MediaPayload yet, since it neither
            # provides InputMedia nor InlineQueryResult.
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
        # Disable the generic extractor to prevent SSRF.
        'allowed_extractors': ('default', '-generic'),
    }

    _instances[audio_only] = ydl = YoutubeDL(cast(Any, opts))
    return ydl


# Serialized to avoid conflicts inside ASSETS_DIR and races on _instances,
# since _executor has max_workers=1.
def ytdlp_task(url: str, audio_only: bool) -> Output:
    log.info('yt-dlp: invoking %s (audio_only=%s)', url, audio_only)
    info = get_ytdlp(audio_only).extract_info(url)

    if is_debug and info:

        def default(o):
            return f'<default: {type(o).__name__}: {o!r}>'

        with open('last_ytdlp_info.json', 'w', encoding='utf-8') as fp:
            json.dump(info, fp, ensure_ascii=False, indent=2, default=default)

    if not info:
        raise RuntimeError('yt-dlp returned no info')

    return Output(cast(dict[str, Any], info))


async def run_ytdlp(url: str, *, audio_only: bool = False) -> Output:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, ytdlp_task, url, audio_only)


async def _handle_yt(
    rs: Responder,
    arg: str,
    cmd: str,
    action: ChatAction,
    *,
    audio_only: bool = False,
    video_note: bool = False,
):
    if not arg:
        await rs.reply_cached(f'Usage: {cmd} <url>')
        return

    if (url := find_url(arg)) is None:
        await rs.reply_cached('Please provide a valid URL.')
        return

    with rs.keep_chat_action(action):
        try:
            output = await run_ytdlp(url, audio_only=audio_only)
            if video_note:
                await output.finish_video_note(rs.get_message())
            else:
                await output.finish(rs, audio_only=audio_only)
        except Exception as e:
            log.exception('ytdlp failed for %s', url)
            error_msg = truncate(f'Download failed: {type(e).__name__}: {e}', 500)
            await rs.reply_cached(error_msg)


@command(public=True)
def handle_yt(rs: Responder, arg: MessageArg):
    return _handle_yt(rs, arg, '/yt', ChatAction.RECORD_VIDEO)


@command(public=True)
def handle_yta(rs: Responder, arg: MessageArg):
    return _handle_yt(rs, arg, '/yta', ChatAction.RECORD_VOICE, audio_only=True)


@command(public=True)
def handle_ytn(rs: Responder, arg: MessageArg):
    return _handle_yt(rs, arg, '/ytn', ChatAction.RECORD_VIDEO_NOTE, video_note=True)


LABEL = {
    'video': '📹 Video',
    'audio': '🎵 Audio',
    'voice': '🎤 Voice',
    'document': '📄 File',
}


def _article(kind: str, url: str, text: str) -> InlineQueryResultArticle:
    return InlineQueryResultArticle(
        id='yt_' + kind,
        title=LABEL[kind],
        input_message_content=InputTextMessageContent(url),
        reply_markup=InlineKeyboardMarkup.from_button(
            InlineKeyboardButton(text, callback_data='noop')
        ),
    )


def handle_yt_inline_query(query: InlineQuery, parsed: tuple[str, str]):
    url, arg = parsed
    log.info('handle_yt_inline_query: %s / %s', url, arg)
    caption = None if 'q' in arg else url

    results = []
    cached_time = 0

    if (voice := _voice_cache.get(url)) is not None:
        payload, title = voice
        results.append(
            payload.as_inline_result(
                caption, id='noop_voice', title='🎤 Voice: ' + title
            )
        )
        if 'voice' in arg:
            return query.answer(results)
        cached_time = None

    if (payload := _media_cache.get(url)) is not None:
        hit_ty = type(payload)
        if hit_ty not in (VideoPayload, AudioPayload, DocumentPayload):
            raise RuntimeError(f'Bad media in cache: {payload}')

        file_name = cast(Media, payload.content).file_name
        title = f'{LABEL[hit_ty.KIND]}: {file_name or url}'
        results.append(payload.as_inline_result(caption, title=title))
        cached_time = None
    else:
        hit_ty = None

    if hit_ty is not VideoPayload:
        results.append(_article('video', url, '📹 Downloading video...'))

    if hit_ty is not AudioPayload:
        results.append(_article('audio', url, '🎵 Downloading audio...'))

    if voice is None:
        results.append(_article('voice', url, '🎤 Encoding voice...'))

    return query.answer(results, cache_time=cached_time)


async def _finish_voice(
    output: Output,
    result: tuple[float, bytes, int],
    url: str,
) -> InlineKeyboardMarkup | None:
    duration, data, bitrate = result
    duration = media_duration(duration)

    if (cached := await VoicePayload(data, duration).stage()) is None:
        return None

    _voice_cache[url] = (cached, output.get_name())
    log.info('Cached voice for %s: %s', url, cached)

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

            async def worker():
                nonlocal markup
                result = await encode_voice(
                    output.path, lambda *_: None, output.duration
                )
                markup = await _finish_voice(output, result, url)
                await bot.edit_message_caption(
                    inline_message_id=inline_message_id,
                    caption=caption,
                    reply_markup=markup,
                )

            create_task(worker())

    except Exception as e:
        log.exception('handle_yt_chosen_result: failed for %s', url)
        error_msg = truncate(f'❌ {type(e).__name__}: {e}', 200)
        await bot.edit_message_caption(
            caption=error_msg, inline_message_id=inline_message_id
        )
        return

    # Upload to staging chat to get file_id, then edit inline message.
    stage_caption = f'{url}\n{get_context().sender} {result_id} {arg}'.strip()
    raw_payload = output._payload(audio_only=audio_only)

    if (payload := await raw_payload.stage(stage_caption)) is not None:
        if (input_media := payload.as_input_now(caption)) is not None:
            if isinstance(payload.content, MEDIA_TYPES):
                _media_cache[url] = payload
            else:
                log.error(f'Bad staged media: {payload}')
            log.info('Media ready: %s', payload)
            return await bot.edit_message_media(
                input_media, inline_message_id=inline_message_id, reply_markup=markup
            )
        log.error('No input_media: %s', payload)

    await bot.edit_message_caption(
        inline_message_id=inline_message_id,
        caption='Failed: ' + url,
        reply_markup=markup,
    )
