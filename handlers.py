import sys

from telegram import (
    Message,
    MessageOriginChannel,
    Update,
    ChosenInlineResult,
    InlineQuery,
)
from telegram.constants import ChatType

from util import (
    log,
    bot,
    pre_block,
    reply_text,
    get_context,
    route_cmd,
    Sender,
    USER_ID,
    CHAN_ID,
    GROUP_ID,
    TODO_ID,
)
from inoue import render_receipt
from rg import handle_rg
from voice import try_handle_voice
from todo import handle_todo_msg
from ytdlp import extract_url, handle_yt_inline_query, handle_yt_chosen_result
from render import handle_render_doc, handle_render_group, handle_render_inline_query
from commands import dispatch_cmd, reply_usage


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
        return await msg.reply_text(*pre_block(str(origin)), do_quote=True)

    if context.sender_is_guest():
        await reply_usage(msg)
        return

    # Privileged operations are only allowed in private chats, even if it's from
    # USER_ID.
    if msg.chat_id != USER_ID:
        # This should not happen since non-private chats are already skipped.
        log.error('handle_msg: unauthorized update: %s', update)
        return

    if not ((text := msg.text) and (text := text.strip())):
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
