import sys
from typing import cast
from pathlib import Path

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
    get_context,
    get_msg,
    get_responder,
    use_msg_override,
    use_responder_override,
    use_text_override,
    route_cmd,
    InlineResponder,
    Sender,
    Context,
    VoicePayload,
    USER_ID,
    CHAN_ID,
    GROUP_ID,
    TODO_ID,
)
from inoue import render_receipt
from rg import handle_rg
from voice import try_handle_voice
from sticker import try_handle_sticker
from todo import handle_todo_msg
from ytdlp import extract_url, handle_yt_inline_query, handle_yt_chosen_result
from render import handle_render_doc, handle_render_group, handle_render_inline_query
from commands import dispatch_cmd, reply_usage
from dispatch import callback_query, dispatch_callback, CallbackData


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

    chat = msg.chat
    if chat.type != ChatType.PRIVATE and (update.message or update.edited_message):
        # For a message update in an ordinary group, we only handle it if a command
        # is matched or we are mentioned, or we will get flooded when we are added
        # as admins.
        if (text := strip_mention(msg)) is None:
            log.debug('Not mentioned: %s', msg)
            return
        with use_text_override(text):
            return await _handle_msg(msg, update, context)

    return await _handle_msg(msg, update, context)


async def _handle_msg(msg: Message, update: Update, context: Context):
    rs = get_responder(msg)

    if await try_handle_voice(msg, rs) or await try_handle_sticker(msg, rs):
        return

    # ID Bot
    if msg.forward_origin:
        return await rs.reply_cached(*pre_block(str(msg.forward_origin)))

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
        and msg.chat_id == USER_ID
        and msg.chat.type == ChatType.PRIVATE
    ):
        log.error('handle_msg: unauthorized update: %s', update)
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
        return await dispatch_cmd(update, rs, text)

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
            await handle_yt_chosen_result(result_id, parsed, result.inline_message_id)
        else:
            log.warning('Invalid chosen inline result: %s', result)
    elif not result_id.startswith('noop'):
        log.error('Bad chosen inline result: %s', result)


def strip_mention(msg: Message) -> str | None:
    # ruff: noqa: E741
    if msg.text:
        text, entities = msg.text, msg.entities
    elif msg.caption:
        text, entities = msg.caption, msg.caption_entities
    else:
        return None

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
                return text


async def handle_guest(msg: Message, update: Update):
    log.debug('handle_guest: %s', msg)
    if (text := strip_mention(msg)) is None:
        raise ValueError(f'missing MENTION entity in guest message: {msg}')

    with (
        use_text_override(text),
        use_responder_override(InlineResponder(msg, answer_guest_query)),
    ):
        await handle_msg(msg, update)


async def handle_callback_query(query: CallbackQuery, update: Update):
    if not (data := query.data) or data == 'noop':
        return query.answer()

    try:
        if (mid := query.inline_message_id) is not None:
            # A guest message created the callback. We continue editing the same
            # inline message, like a continuation of `handle_guest`.

            # We use `CallbackQuery` as a minimal stub to provide `Message.from_user`
            # and give `None` for everything else, which is enough for dispatching
            # until `route_cmd()`.
            msg = cast(Message, CallbackQueryAsMessage(query))
            rs = InlineResponder(msg, mid)
            log.debug('inline callback: %r', query)
            with use_msg_override(msg), use_responder_override(rs):
                await dispatch_callback(data, update)
        else:
            await dispatch_callback(data, update)
    except Exception as e:
        await query.answer('Error', show_alert=True)
        raise e


class CallbackQueryAsMessage:
    __slots__ = ('_query',)

    def __init__(self, query: CallbackQuery):
        self._query = query

    def __getattr__(self, item: str):
        r = getattr(self._query, item, None)
        log.debug('CallbackQueryAsMessage: getattr %r -> %r', item, r)
        return r

    def __repr__(self) -> str:
        return f'CallbackQueryAsMessage({self._query!r})'


@callback_query('relay')
async def handle_relay_callback(
    query: CallbackQuery, data: CallbackData, update: Update
):
    text = data[data.index('_') + 1 :]
    msg = get_msg(update)
    log.debug('relay: %r %s', msg, text)
    with use_text_override(text):
        await handle_msg(msg, update)
    await query.answer()
