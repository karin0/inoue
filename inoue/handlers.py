import asyncio
import sys

from datetime import datetime
from pathlib import Path

from telegram import (
    CallbackQuery,
    Chat,
    ChosenInlineResult,
    InlineQuery,
    Message,
    MessageOriginChannel,
    User,
)
from telegram.constants import ChatType, MessageEntityType

from bot import CallbackData, Responder, VoicePayload, bot, callback_query

from .commands import dispatch_cmd, handle_help
from .ctx import Sender, get_context, is_admin
from .env import CHAN_ID, GROUP_ID, TODO_ID, USER_ID
from .inoue import render_receipt
from .log import log
from .render import handle_render_doc, handle_render_group, handle_render_inline_query
from .rg import handle_rg
from .sticker import handle_sticker
from .text import pre_block
from .todo import handle_todo_msg
from .voice import handle_voice
from .ytdlp import extract_url, handle_yt_chosen_result, handle_yt_inline_query

'''
Responders are only present when handling the following events, and shall not be
used on other paths:

- message
- edit_message
- callback_query (with an accessible message or inline message)
- chosen_inline_result
- guest_message

For callback queries with an inline message and `guest_message`, `InlineResponder`
is used to enable (almost) seamless replying by editing the same inline message.

For callback queries with an `InaccessibleMessage`, we pass `None` to the dispatcher,
so exceptions may be raised by the `Route` if the handler requires it.

Technically we could create a `Responder` for channel posts and inline queries,
but we don't think they should be responded to for now.
'''


def handle_post(channel_post: Message, sender: Sender | None):
    if sender is not None and sender.id in (USER_ID, CHAN_ID):
        return handle_render_doc(channel_post)


async def handle_msg(rs: Responder, direct: bool = True):
    context = get_context()
    msg = rs.get_message()
    log.debug('handle_msg: rs: %s, sender: %s', rs, context.sender)

    if TODO_ID and msg.chat_id == TODO_ID:
        # Responders are not used for messages in `TODO_ID`, which has a different
        # interaction model.
        if context.sender_is_host():
            return await handle_todo_msg(msg)
        raise ValueError(f'Unauthorized todo: {msg}')

    # `render` handles Doc messages that are forwarded from CHAN_ID to its discussion group.
    if (
        isinstance(origin := msg.forward_origin, MessageOriginChannel)
        and msg.is_automatic_forward
        and msg.chat_id == GROUP_ID
        and origin.chat.id == CHAN_ID
    ):
        return await handle_render_group(rs, origin.message_id)

    # Reroute if the text starts with a command.
    # This allows commands to be sent as any styled text (pre/quote), rather than
    # a canonical BOT_COMMAND entity.
    if (coro := rs.dispatch_command()) is not None:
        return await coro

    if (
        msg.chat.type != ChatType.PRIVATE
        and direct
        and not (
            strip_mention(rs)
            or (m := msg.reply_to_message) is not None
            and (u := m.from_user) is not None
            and u.id == bot.bot.id
        )
    ):
        # For a message update in an ordinary group, we only handle it if a command is matched,
        # we are mentioned, or we are replied to. Otherwise we will get flooded when we are added
        # as admins.
        log.debug('Not mentioned: %s', msg)
        return None

    if msg.via_bot is not None:
        log.debug('Via bot: %s', msg)
        return None

    if await handle_voice.route(rs, as_command=False) or await handle_sticker(
        msg, rs, as_command=False
    ):
        return None

    # ID Bot
    if origin is not None:
        return await rs.reply_cached(*pre_block(str(origin)))

    if not (text := rs.get_text().strip()):
        if (
            msg.forum_topic_created
            or msg.forum_topic_edited
            or msg.forum_topic_closed
            or msg.forum_topic_reopened
        ):
            log.debug('Ignoring forum_topic: %s', msg)
            return None

        return await rs.reply(media=VoicePayload(Path('out.ogg')))

    # We always check the sender's identity from the `context` rather than `msg`
    # itself, since it could be a mocked one from relayed callback queries or
    # guest messages.
    if context.sender_is_guest():
        return await handle_help(rs)

    # Administration is only allowed for the host in their own private chat.
    if not (context.sender_is_host() and is_admin(context.update, msg)):
        log.error('handle_msg: unauthorized update: %s', context)
        return None

    if text == '/Please log out now/':
        # Be careful, since you won't be able to log in again within 10 minutes.
        await rs.reply_cached('See you next time!')
        if await bot.log_out():
            log.info('log_out: success')
            sys.exit(1)
        else:
            log.error('log_out: failed')
        return None

    if text.startswith('/'):
        return await dispatch_cmd(rs)

    if '\n' not in text:
        return await handle_rg(rs, text)

    await rs.reply_cached(*pre_block(render_receipt(text)))


async def handle_inline_query(query: InlineQuery):
    if data := query.query.strip():
        if (parsed := extract_url(data)) is not None:
            await handle_yt_inline_query(query, parsed)
        else:
            await handle_render_inline_query(query, data)


async def handle_chosen_inline(rs: Responder | None, result: ChosenInlineResult):
    result_id = result.result_id
    if result_id.startswith('yt_'):
        if rs is not None and (parsed := extract_url(result.query.strip())) is not None:
            await handle_yt_chosen_result(result_id, parsed, rs)
        else:
            log.warning('Invalid chosen inline result: %s', result)
    elif not result_id.startswith('noop'):
        log.error('Bad chosen inline result: %s', result)


def strip_mention(rs: Responder) -> bool:
    # ruff: noqa: E741
    msg = rs.get_message()
    if msg.text:
        text, entities = msg.text, msg.entities
    elif msg.caption:
        text, entities = msg.caption, msg.caption_entities
    else:
        return False

    text = text.encode('utf-16-le')
    for ent in entities:
        if ent.type == MessageEntityType.MENTION:
            l = ent.offset << 1
            r = (ent.offset + ent.length) << 1
            part = text[l:r].decode('utf-16-le')
            if part == '@' + bot.username:
                left = text[:l].decode('utf-16-le').rstrip()
                right = text[r:].decode('utf-16-le').lstrip()
                text = (left + ' ' + right).strip()
                log.info('strip_mention: extracted text: %s', text)
                rs.set_text(text)
                return True
    return False


def handle_guest(rs: Responder):
    log.debug('handle_guest: %s', rs)
    # Guest messages may mention or reply to us.
    strip_mention(rs)
    return handle_msg(rs, direct=False)


async def handle_callback_query(rs: Responder | None, query: CallbackQuery):
    log.debug('inline callback: %r', query)
    if not (data := query.data) or data == 'noop':
        return await query.answer()

    try:
        if rs is not None and (coro := rs.dispatch_callback_query(data)) is not None:
            return await coro
        log.warning('Bad callback query: %s, %s', data, rs)
    except Exception:
        await query.answer('Error', show_alert=True)
        raise


def message_stub(from_user: User) -> Message:
    return Message(0, datetime.now(), Chat(0, ChatType.SENDER), from_user=from_user, text='')


@callback_query('relay')
def handle_relay_callback(query: CallbackQuery, data: CallbackData, rs: Responder):
    text = data[data.index('_') + 1 :]
    log.debug('relay: %s', text)
    rs.set_text(text)
    return asyncio.gather(handle_msg(rs, direct=False), query.answer())
