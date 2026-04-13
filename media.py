from telegram import Message, Update
from telegram.ext import ContextTypes

from db import db
from util import (
    log,
    escape,
    get_msg,
    get_msg_arg,
    get_msg_url,
    get_deep_link_url,
    reply_text,
)


def extract_media(msg: Message) -> tuple[str, str] | None:
    if (
        f := msg.voice
        or msg.document
        or msg.audio
        or msg.video
        or msg.animation
        or msg.sticker
        or (msg.photo[0] if msg.photo else None)
    ):
        file_name = getattr(f, 'file_name', '')
        mime_type = getattr(f, 'mime_type', None)
        log.info(
            'Saving media: chat=%s, message=%s, name=%s, mime=%s, size=%s',
            msg.chat_id,
            msg.message_id,
            file_name or '<unnamed>',
            mime_type or '<unknown>',
            f.file_size,
        )
        return file_name, f.file_id

    return None


async def handle_save(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg, arg = get_msg_arg(update)

    if (target := msg.reply_to_message) and (info := extract_media(target)) is not None:
        pass
    elif (info := extract_media(msg)) is not None:
        target = msg
    else:
        return await reply_text(
            msg, r'Send or reply to a media message with `/save [title]` to save it\.'
        )

    file_name, file_id = info
    title = arg.strip() or file_name or ''
    is_new = db.save_media(target.chat_id, target.message_id, title, file_id)
    media_text = render_media(target.chat_id, target.message_id, title)
    if is_new:
        text = rf'Saved {media_text}'
    else:
        text = rf'Updated {media_text}'
    await reply_text(msg, text, 'MarkdownV2')


def render_title(title: str) -> str:
    return f'*{escape(title)}*' if title else '_Untitled_'


def render_ids(chat_id: int, message_id: int) -> str:
    return f'`{escape(str(chat_id))}/{message_id}`'


def render_media(chat_id: int, message_id: int, title: str) -> str:
    return f'{render_title(title)}: {render_ids(chat_id, message_id)}'


async def handle_play(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = get_msg(update)
    if not (media := db.random_media()):
        return await msg.reply_text('No saved media.', do_quote=True)

    chat_id, message_id = media
    log.info('Forwarding saved media: %s/%s -> %s', chat_id, message_id, msg.chat_id)
    await msg.reply_copy(chat_id, message_id, do_quote=True)


async def handle_playlist(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = get_msg(update)
    items = list(db.iter_media())
    if not items:
        return await msg.reply_text('No saved media.', do_quote=True)

    lines = []
    for i, (chat_id, message_id, title, _) in enumerate(items, 1):
        msg_url = get_msg_url(message_id, chat_id)
        play_url = get_deep_link_url(f'play_{chat_id}_{message_id}')
        unsave_url = get_deep_link_url(f'unsave_{chat_id}_{message_id}')
        title_text = render_title(title)
        lines.append(
            f'{escape(f'{i}.')} [{title_text}]({msg_url}) \\| [play]({play_url}) \\| [remove]({unsave_url})'
        )

    await msg.reply_text(
        '\n'.join(lines),
        'MarkdownV2',
        do_quote=True,
        disable_web_page_preview=True,
    )


async def handle_play_media(msg: Message, chat_id: int, message_id: int):
    if not db.has_media(chat_id, message_id):
        return await reply_text(msg, 'Media not found.')

    return await msg.get_bot().forward_message(
        msg.chat_id,
        chat_id,
        message_id,
        message_thread_id=msg.message_thread_id,
    )


async def handle_remove_media(msg: Message, chat_id: int, message_id: int):
    title = db.delete_media(chat_id, message_id)
    if title is not None:
        return await reply_text(
            msg,
            f'Removed {render_media(chat_id, message_id, title)}',
            'MarkdownV2',
        )
    return await reply_text(msg, f'Media not found.')
