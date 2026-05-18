import sys
import asyncio
from typing import cast
from pathlib import Path

from telegram import (
    Message,
    MessageOriginChannel,
    ChosenInlineResult,
    InlineQuery,
    CallbackQuery,
)
from telegram.constants import ChatType, MessageEntityType

from bot import (
    bot,
    Responder,
    InlineResponder,
    VoicePayload,
    callback_query,
    dispatch_callback,
    CallbackData,
)

from .future import answer_guest_query
from .log import log
from .ctx import get_context, Sender
from .env import USER_ID, CHAN_ID, GROUP_ID, TODO_ID
from .text import pre_block
from .inoue import render_receipt
from .rg import handle_rg
from .voice import handle_voice
from .sticker import handle_sticker
from .todo import handle_todo_msg
from .ytdlp import extract_url, handle_yt_inline_query, handle_yt_chosen_result
from .render import handle_render_doc, handle_render_group, handle_render_inline_query
from .commands import dispatch_cmd, reply_usage

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


async def handle_msg(msg: Message, rs: Responder | None = None, direct: bool = True):
    if rs is None:
        rs = Responder.create(msg)

    context = get_context()
    log.debug('handle_msg: rs: %s, sender: %s', rs, context.sender)

    if msg.chat_id == TODO_ID:
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
    if (handler := rs.get_route()) is not None:
        return await handler(rs)

    chat = msg.chat
    if chat.type != ChatType.PRIVATE and direct and not strip_mention(rs):
        # For a message update in an ordinary group, we only handle it if a command
        # is matched or we are mentioned, or we will get flooded when we are added
        # as admins.
        log.debug('Not mentioned: %s', msg)
        return

    if msg.via_bot is not None:
        log.debug('Via bot: %s', msg)
        return

    if await handle_voice.route(rs, as_command=False) or await handle_sticker(
        msg, rs, as_command=False
    ):
        return

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
            return

        return await rs.reply(media=VoicePayload(Path('out.ogg')))

    # We always check the sender's identity from the `context` rather than `msg`
    # itself, since it could be a mocked one from relayed callback queries or
    # guest messages.
    if context.sender_is_guest():
        return await reply_usage(rs)

    # Administration is only allowed for the host in their own private chat.
    if not (
        context.sender_is_host()
        and chat.id == USER_ID
        and chat.type == ChatType.PRIVATE
    ):
        log.error('handle_msg: unauthorized update: %s', context)
        return

    if text == '/Please log out now/':
        # Be careful, since you won't be able to log in again within 10 minutes.
        await rs.reply_cached('See you next time!')
        if await bot.log_out():
            log.info('log_out: success')
            sys.exit(1)
        else:
            log.error('log_out: failed')
        return

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


async def handle_chosen_inline(result: ChosenInlineResult):
    result_id = result.result_id
    if result_id.startswith('yt_'):
        query = result.query.strip()
        if result.inline_message_id and (parsed := extract_url(query)) is not None:
            rs = InlineResponder(MessageStub.cast(result), result.inline_message_id)
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


def handle_guest(msg: Message):
    log.debug('handle_guest: %s', msg)
    rs = InlineResponder(msg, answer_guest_query)
    if strip_mention(rs):
        return handle_msg(msg, rs, direct=False)

    raise ValueError(f'missing MENTION entity in guest message: {msg}')


async def handle_callback_query(query: CallbackQuery):
    if not (data := query.data) or data == 'noop':
        return query.answer()

    try:
        if not isinstance(msg := query.message, Message):
            log.debug('inline callback: %r', query)
            if (mid := query.inline_message_id) is not None:
                # A guest message created the callback. We continue editing the same
                # inline message, like a continuation of `handle_guest`.
                rs = InlineResponder(MessageStub.cast(msg or query), mid)
            else:
                rs = None
        else:
            rs = Responder.create(msg)
        await dispatch_callback(rs, data)
    except Exception as e:
        await query.answer('Error', show_alert=True)
        raise e


# We use `InaccessibleMessage`, `CallbackQuery` or `ChosenInlineResult` as a
# minimal stub to provide `Message.from_user` and give `None` for everything else,
# which is enough for dispatching until `get_route()`.
class MessageStub:
    __slots__ = ('_data',)

    def __init__(self, data):
        self._data = data

    @staticmethod
    def cast(data) -> Message:
        return cast(Message, MessageStub(data))

    def __getattr__(self, item: str):
        r = getattr(self._data, item, None)
        log.debug('MessageStub: getattr %r -> %r', item, r)
        return r

    def __repr__(self) -> str:
        return f'MessageStub({self._data!r})'


@callback_query('relay')
def handle_relay_callback(
    query: CallbackQuery, data: CallbackData, msg: Message, rs: Responder
):
    text = data[data.index('_') + 1 :]
    log.debug('relay: %r %s', msg, text)
    rs.set_text(text)
    return asyncio.gather(handle_msg(msg, rs, direct=False), query.answer())
