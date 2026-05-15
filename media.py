from telegram import Message

from db import db
from util import log, escape, get_msg_url, get_deep_link_url, reply_text, Responder
from dispatch import MessageArg, command, start


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


@command
def handle_save(msg: Message, arg: MessageArg):
    if (target := msg.reply_to_message) and (info := extract_media(target)) is not None:
        pass
    elif (info := extract_media(msg)) is not None:
        target = msg
    else:
        return reply_text(
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
    return reply_text(msg, text, 'MarkdownV2')


def render_title(title: str) -> str:
    return f'*{escape(title)}*' if title else '_Untitled_'


def render_ids(chat_id: int, message_id: int) -> str:
    return f'`{escape(str(chat_id))}/{message_id}`'


def render_media(chat_id: int, message_id: int, title: str) -> str:
    return f'{render_title(title)}: {render_ids(chat_id, message_id)}'


@command(public=True)
def handle_play(rs: Responder, arg: MessageArg):
    if arg:
        if not (media := db.get_media(int(arg))):
            return rs.reply_cached('Media not found.')
    elif not (media := db.random_media()):
        return rs.reply_cached('No saved media.')

    chat_id, message_id = media
    log.info('Forwarding saved media: %s/%s -> %s', chat_id, message_id, rs)
    return rs.reply_copy(chat_id, message_id)


def _format_item(row: tuple[int, int, int, str, int]) -> str:
    id, chat_id, message_id, title, unix = row
    msg_url = get_msg_url(message_id, chat_id)
    play_url = get_deep_link_url(f'play_{id}')
    unsave_url = get_deep_link_url(f'unsave_{id}')
    title_text = render_title(title)
    return (
        f'{escape(f'{id}.')} [{title_text}]({msg_url}) '
        f'\\| ![{unix}](tg://time?unix={unix}&format=DT) '
        f'\\| [play]({play_url}) \\| [remove]({unsave_url})'
    )


@command
def handle_playlist(msg: Message):
    if not (text := '\n'.join(map(_format_item, db.iter_media()))):
        return reply_text(msg, 'No saved media.')

    return reply_text(msg, text, 'MarkdownV2', disable_web_page_preview=True)


@start('play')
def handle_play_media(msg: Message, id: int):
    if (media := db.get_media(id)) is None:
        return reply_text(msg, 'Media not found.')

    return msg.get_bot().forward_message(
        msg.chat_id,
        *media,
        message_thread_id=msg.message_thread_id,
    )


@start('unsave')
def handle_remove_media(msg: Message, id: int):
    if (row := db.delete_media(id)) is None:
        return reply_text(msg, 'Media not found.')

    return reply_text(
        msg,
        f'Removed {render_media(*row)}',
        'MarkdownV2',
    )
