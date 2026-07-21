import asyncio
import os

from collections.abc import Awaitable
from io import BytesIO
from pathlib import Path
from typing import cast

from pathvalidate import sanitize_filename
from PIL import Image
from telegram import Animation, Document, Message, PhotoSize, Sticker
from telegram.constants import ChatAction

from bot import DocumentPayload, Responder, bot, command

from .ffmpeg import capture_ffmpeg
from .log import log

STICKER_SIDE = 512
MAX_FILE_SIZE = 10 << 20
MAX_WEBM_SIZE = 256_000

LOTTIE_TO_GIF = 'lottie_to_gif.sh'
ASSETS_DIR = 'assets/sticker'

WEBM_ARGS = (
    '-vf',
    f'scale=w={STICKER_SIDE}:h={STICKER_SIDE}:force_original_aspect_ratio=decrease',
    '-c:v',
    'libvpx-vp9',
    '-f',
    'webm',
    '-an',
    'pipe:1',
)


def to_webp(src: str, rs: Responder) -> bytes:
    img = Image.open(src)
    log.info('to_webp: %s, %d x %d', img.format, *img.size)
    img.thumbnail((STICKER_SIDE, STICKER_SIDE), Image.Resampling.LANCZOS)
    if img.mode not in ('RGBA', 'RGB', 'P'):
        img = img.convert('RGBA')

    if 'rembg' in rs.get_text():
        log.info('to_webp: invoking rembg!')
        from rembg import remove

        img = cast(Image.Image, remove(img))

    buf = BytesIO()
    img.save(buf, 'webp', lossless=True)
    return buf.getvalue()


async def to_webm(src: str) -> bytes:
    base = ('-t', '3', '-i', src)
    data = await capture_ffmpeg(*base, '-lossless', '1', *WEBM_ARGS, desc='webm/lossless')
    if len(data) <= MAX_WEBM_SIZE:
        return data
    log.info('Lossless webm too large (%d B), retrying lossy', len(data))
    return await capture_ffmpeg(*base, *WEBM_ARGS, desc='webm/lossy')


# Transparent animated GIF: reserve a palette slot for transparency and map
# low-alpha source pixels onto it. `-transdiff` disables GIF frame-diff
# transparency, which otherwise ghosts the real alpha into a blotchy
# background. bayer dithering keeps flat sticker art free of the noise that
# error-diffusion (the default) sprays across solid regions.
GIF_FILTER = (
    'split[a][b];'
    '[a]palettegen=reserve_transparent=1:stats_mode=full[p];'
    '[b][p]paletteuse=alpha_threshold=128:dither=bayer:bayer_scale=3'
)


async def webm_to_gif(src: str) -> bytes:
    return await capture_ffmpeg(
        # Telegram video stickers carry alpha as a VP9 side-channel that ffmpeg's
        # native decoder silently drops (compositing onto black). Only libvpx-vp9
        # decodes it, so the filter graph below actually sees the transparency.
        '-c:v',
        'libvpx-vp9',
        '-i',
        src,
        '-filter_complex',
        GIF_FILTER,
        '-gifflags',
        '-transdiff',
        '-f',
        'gif',
        'pipe:1',
        desc='webm/gif',
    )


async def tgs_to_gif(path: str) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        LOTTIE_TO_GIF,
        path,
        '--output',
        '-',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    if proc.returncode:
        raise RuntimeError(f'lottie_to_gif failed: {err.decode(errors="replace")}')
    return out


class TooLarge(ValueError):
    pass


async def download(media: Sticker | PhotoSize | Animation | Document, ext: str) -> str:
    if (media.file_size or 0) > MAX_FILE_SIZE:
        raise TooLarge

    f = await bot.get_file(media.file_id)
    if f.file_path and os.path.isfile(f.file_path):
        log.info('sticker: Local: %s', os.path.basename(f.file_path))
        return f.file_path

    if file_name := getattr(media, 'file_name', f.file_path):
        ext = os.path.splitext(file_name)[1].lower() or ext

    dst = os.path.join(ASSETS_DIR, sanitize_filename(media.file_unique_id + ext))
    if os.path.isfile(dst):
        log.info('sticker: Cached: %s', dst)
        return dst

    os.makedirs(ASSETS_DIR, exist_ok=True)
    await f.download_to_drive(dst)
    log.info('sticker: Downloaded: %s', dst)
    return dst


async def send(
    rs: Responder,
    data: bytes | str | Awaitable[bytes],
    base: str | None,
    ext: str,
    caption: str | None = None,
    raw: bool = False,
):
    name = (base and os.path.splitext(base)[0] or 'out') + ext
    content: bytes | Path
    if isinstance(data, str):
        content = Path(data)
        log.info('sticker: send: %s (%s)', name, content.name)
    elif isinstance(data, bytes):
        content = data
        log.info('sticker: send: %s (%d B)', name, len(data))
    else:
        content = await cast(Awaitable[bytes], data)
        log.info('sticker: send: %s (%d B)', name, len(content))

    return await rs.reply(
        caption, media=DocumentPayload(content, filename=name, disable_content_type_detection=raw)
    )


async def _handle_sticker(rs: Responder, src: Message) -> str | None:
    if sti := src.sticker:
        base, emoji = sti.set_name, sti.emoji
        if sti.is_video:
            with rs.keep_chat_action(ChatAction.UPLOAD_VIDEO):
                path = await download(sti, '.webm')
                await asyncio.gather(
                    send(rs, path, base, '.webm', emoji, raw=True),
                    send(rs, webm_to_gif(path), base, '.gif', emoji, raw=True),
                )
        elif sti.is_animated:
            with rs.keep_chat_action(ChatAction.UPLOAD_VIDEO):
                path = await download(sti, '.tgs')
                await send(rs, tgs_to_gif(path), base, '.gif', emoji, raw=True)
        else:
            with rs.keep_chat_action(ChatAction.UPLOAD_PHOTO):
                path = await download(sti, '.webp')
                await send(rs, path, base, '.webp', emoji, raw=True)
        return path

    if photos := src.photo:
        ph = next(
            (p for p in photos if p.width >= STICKER_SIDE or p.height >= STICKER_SIDE), photos[-1]
        )
        with rs.keep_chat_action(ChatAction.CHOOSE_STICKER):
            path = await download(ph, '.jpg')
            await send(rs, to_webp(path, rs), None, '.webp')
        return path

    if ani := src.animation:
        with rs.keep_chat_action(ChatAction.UPLOAD_VIDEO):
            path = await download(ani, '.mp4')
            await send(rs, to_webm(path), ani.file_name, '.webm')
        return path

    if doc := src.document:
        mime = doc.mime_type or ''
        name = doc.file_name or ''
        ext = os.path.splitext(name)[1].lower()
        if ext == '.gif' or mime.startswith('video'):
            with rs.keep_chat_action(ChatAction.UPLOAD_VIDEO):
                path = await download(doc, ext)
                await send(rs, to_webm(path), name, '.webm')
            return path
        if mime.startswith('image'):
            with rs.keep_chat_action(ChatAction.CHOOSE_STICKER):
                path = await download(doc, ext)
                await send(rs, to_webp(path, rs), name, '.webp')
            return path


@command(public=True)
async def handle_sticker(rs: Responder, as_command: bool = True) -> bool:
    try:
        r = await rs.extract_async(_handle_sticker)
    except TooLarge:
        await rs.reply_cached('File is too large.')
        return True
    if r is None and as_command:
        await rs.reply_cached('Send or reply to a photo/animation to convert it into a sticker.')
    return r is not None
