import sys
from typing import cast

from telegram import (
    Message,
    MessageOriginChannel,
    Update,
    ChosenInlineResult,
    InlineQuery,
    CallbackQuery,
)
from telegram.constants import ChatType, MessageEntityType

from future import answer_guest_query
from util import (
    log,
    bot,
    pre_block,
    reply_text,
    get_context,
    get_text,
    use_msg_override,
    route_cmd,
    Sender,
    USER_ID,
    CHAN_ID,
    GROUP_ID,
    TODO_ID,
    InlineMessageProxy,
)
from inoue import render_receipt
from rg import handle_rg
from voice import try_handle_voice
from todo import handle_todo_msg
from ytdlp import extract_url, handle_yt_inline_query, handle_yt_chosen_result
from render import handle_render_doc, handle_render_group, handle_render_inline_query
from commands import dispatch_cmd, reply_usage
from dispatch import callback_query, CallbackData


def handle_post(channel_post: Message, sender: Sender | None):
    if sender is not None and sender.id in (USER_ID, CHAN_ID):
        return handle_render_doc(channel_post)


async def handle_msg(msg: Message, update: Update):
    context = get_context()
    log.debug('handle_msg: sender: %s', context.sender)

    if msg.chat_id == TODO_ID:
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
        return await handle_render_group(msg, origin.message_id)

    # Reroute if the text starts with a command.
    # This allows commands to be sent as any styled text (pre/quote), rather than
    # a canonical BOT_COMMAND entity.
    if (fut := route_cmd(update, msg)) is not None:
        return await fut

    if msg.chat.type != ChatType.PRIVATE:
        return

    if await try_handle_voice(msg):
        return

    # ID Bot
    if origin:
        return await reply_text(msg, *pre_block(str(origin)))

    if not (text := get_text(msg).strip()):
        if (
            msg.forum_topic_created
            or msg.forum_topic_edited
            or msg.forum_topic_closed
            or msg.forum_topic_reopened
        ):
            log.debug('Ignoring forum_topic: %s', msg)
            return

        with open('out.ogg', 'rb') as f:
            return await msg.reply_voice(f, do_quote=True)

    # Warning: Always check the sender's identity from the `context` rather than
    # `msg` itself, since it could be a mocked one from relayed callback queries
    # or guest messages.
    if context.sender_is_guest():
        await reply_usage(msg)
        return

    if not context.sender_is_host():
        log.error('handle_msg: unauthorized update: %s', update)
        return

    # Be careful, since you won't be able to log in again within 10 minutes.
    if text == '/Please log out now/':
        await reply_text(msg, 'See you next time!')
        if await bot.log_out():
            log.info('log_out: success')
            sys.exit(1)
        else:
            log.error('log_out: failed')
        return

    if text.startswith('/'):
        return await dispatch_cmd(update, msg, text)

    if '\n' not in text:
        return await handle_rg(msg, text)

    await reply_text(msg, *pre_block(render_receipt(text)))


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
            await handle_yt_chosen_result(result_id, parsed, result.inline_message_id)
        else:
            log.warning('Invalid chosen inline result: %s', result)
    elif not result_id.startswith('noop'):
        log.error('Bad chosen inline result: %s', result)


async def handle_guest(msg: Message, update: Update):
    # ruff: noqa: E741
    log.debug('handle_guest: %s', msg)
    assert msg.text and msg.entities

    text = msg.text.encode('utf-16-le')
    for ent in msg.entities:
        if ent.type == MessageEntityType.MENTION:
            l = ent.offset << 1
            r = (ent.offset + ent.length) << 1
            part = text[l:r].decode('utf-16-le')
            if part == '@' + bot.username:
                left = text[:l].decode('utf-16-le').rstrip()
                right = text[r:].decode('utf-16-le').lstrip()
                text = (left + ' ' + right).strip()
                log.info('handle_guest: extracted text: %s', text)
                break
    else:
        raise ValueError(f'missing MENTION entity in guest message: {msg}')

    msg = cast(Message, InlineMessageProxy(msg, answer_guest_query))
    with use_msg_override(msg, text):
        await handle_msg(msg, update)


@callback_query('relay')
async def handle_relay_callback(
    query: CallbackQuery, data: CallbackData, update: Update
):
    text = data[data.index('_') + 1 :]
    msg = query.message

    if not isinstance(msg, Message):
        if (mid := query.inline_message_id) is not None:
            # A guest message created the callback with `/run`.
            # We will edit our answer to the guest query with `InlineMessageProxy`,
            # which behaves like a continuation of `handle_guest`.
            msg = cast(Message, InlineMessageProxy(query, mid))
        else:
            raise ValueError(f'No message in callback query: {query}')

    log.debug('relay: %r %s', msg, text)
    with use_msg_override(msg, text):
        await handle_msg(msg, update)

    await query.answer()
