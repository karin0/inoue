import os
import sys
import atexit
import asyncio

from telegram import Update, Bot, MessageOriginChannel
from telegram.ext import (
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    ChosenInlineResultHandler,
    InlineQueryHandler,
)
from telegram.constants import ChatType

from util import (
    log,
    is_debug,
    app,
    post_init,
    pre_block,
    reply_text,
    ME,
    USER_ID,
    CHAN_ID,
    GROUP_ID,
    TODO_ID,
    LOCK_FILE,
    get_context,
    get_msg,
    do_notify,
    try_reroute_cmd,
)
from gateway import add_handler, add_command_handler
from dispatch import handle_callback_query, iter_commands
from inoue import render_receipt
from rg import handle_rg
from voice import try_handle_voice
from todo import handle_todo_msg
from ytdlp import extract_url, handle_yt_inline_query, handle_yt_chosen_result
from render import handle_render_doc, handle_render_group, handle_render_inline_query
from commands import dispatch_cmd, set_commands, stats, reply_usage

import misc, media, run  # noqa: F401, E401


async def handle_msg(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    context = get_context()
    log.debug('handle_msg: sender: %s', context.sender)
    if (
        (sender := context.sender) is not None
        and sender.id in (USER_ID, CHAN_ID)
        and (post := update.edited_channel_post or update.channel_post)
    ):
        return await handle_render_doc(update, post)

    if not (msg := get_msg(update)):
        return

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
    if (fut := try_reroute_cmd(update, ctx, msg)) is not None:
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
        if await ctx.bot.log_out():
            log.info('log_out: success')
            sys.exit(1)
        else:
            log.error('log_out: failed')
        return

    if text.startswith('/'):
        return await dispatch_cmd(update, ctx, msg, text)

    if '\n' not in text:
        return await handle_rg(msg, text)

    await reply_text(msg, *pre_block(render_receipt(text)))


async def handle_inline_query(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.inline_query
    assert query
    if data := query.query.strip():
        if (parsed := extract_url(data)) is not None:
            await handle_yt_inline_query(query, parsed)
        else:
            await handle_render_inline_query(update, ctx, query, data)


async def handle_chosen_inline(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    result = update.chosen_inline_result
    assert result
    result_id = result.result_id
    if result_id.startswith('yt_'):
        query = result.query.strip()
        if result.inline_message_id and (parsed := extract_url(query)) is not None:
            await handle_yt_chosen_result(
                ctx.bot,
                result_id,
                parsed,
                result.inline_message_id,
            )
        else:
            log.warning('Invalid chosen inline result: %s', result)
    elif not result_id.startswith('noop'):
        log.error('Bad chosen inline result: %s', result)


def _post_init(bot: Bot):
    log.info('%s initiated: %s', ME, bot.bot)
    return asyncio.gather(
        set_commands(bot),
        do_notify(*stats(bot.bot, f'{ME} initiated'), quiet=is_debug),
    )


def init_app():
    post_init(_post_init)

    for name, (func, permissive) in iter_commands():
        if name == 'rg':
            names = (name, *(f'{name}{off}' for off in range(0, 5)))
        else:
            names = (name,)
        add_command_handler(app, names, func, permissive=permissive)

    add_handler(app, MessageHandler, True, None, handle_msg)
    add_handler(app, CallbackQueryHandler, True, handle_callback_query)
    add_handler(app, InlineQueryHandler, True, handle_inline_query)
    add_handler(app, ChosenInlineResultHandler, True, handle_chosen_inline)


def get_lock():
    fd = os.open(LOCK_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, 'w') as fp:
        fp.write(str(os.getpid()))
    atexit.register(drop_lock)


def drop_lock():
    try:
        os.remove(LOCK_FILE)
    except OSError as e:
        print(f'Failed to remove lock: {e}', file=sys.stderr)
    atexit.unregister(drop_lock)


def main():
    log.info('Starting %s...', ME)
    get_lock()
    init_app()
    app.run_polling()
    log.info('%s stopped.', ME)
    drop_lock()


if __name__ == '__main__':
    main()
