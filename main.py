import os
import sys
import atexit
import asyncio

from telegram import Update, Bot, MessageOriginChannel
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    ChosenInlineResultHandler,
    InlineQueryHandler,
    Application,
)
from telegram.error import NetworkError
from telegram.constants import ChatType

from util import (
    log,
    is_debug,
    notify,
    pre_block,
    reply_text,
    USER_ID,
    CHAN_ID,
    GROUP_ID,
    TODO_ID,
    DB_FILE,
    LOCK_FILE,
    init_util,
    get_msg,
    do_notify,
)
from db import db
from gateway import add_handler, add_command_handler
from context import ME, get_sender
from dispatch import handle_callback_query, iter_commands
from inoue import render_receipt
from rg import handle_rg
from voice import try_handle_voice
from todo import handle_todo_msg
from ytdlp import extract_url, handle_yt_inline_query, handle_yt_chosen_result
from render import handle_render_doc, handle_render_group, handle_render_inline_query
from commands import dispatch_cmd, set_commands, stats, reply_usage

import misc, media, run


async def handle_msg(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    sender = get_sender()
    is_guest = bool(sender and sender.is_guest)
    if not is_guest and (post := update.edited_channel_post or update.channel_post):
        return await handle_render_doc(update, post)

    if not (msg := get_msg(update)):
        return

    if msg.chat_id == TODO_ID:
        if sender and sender.id == USER_ID:
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

    if msg.chat.type != ChatType.PRIVATE:
        return

    if await try_handle_voice(msg):
        return

    # ID Bot
    if origin:
        return await msg.reply_text(*pre_block(str(origin)), do_quote=True)

    if is_guest:
        assert sender is not None
        await reply_usage(msg, sender)
        return

    # Privileged operations are only allowed in private chats, even if it's from
    # USER_ID.
    if msg.chat_id != USER_ID:
        # This should not happen since non-private chats are already skipped.
        log.error('handle_msg: unauthorized update: %s', update)
        return

    if not ((text := msg.text) and text.strip()):
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

    # Be more careful, since you won't be able to log in again within 10 minutes.
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


async def post_init(app: Application) -> None:
    bot: Bot = app.bot
    init_util(bot)
    db.connect(DB_FILE)
    log.info('%s initiated: %s', ME, bot.bot)
    await asyncio.gather(
        set_commands(bot),
        do_notify(*stats(bot.bot, f'{ME} initiated'), quiet=is_debug),
    )


async def post_stop(_: Application) -> None:
    db.close()


async def handle_error(update, context: ContextTypes.DEFAULT_TYPE) -> None:
    e = context.error
    if isinstance(e, NetworkError):
        with notify.suppress():
            log.error('Network error in update %s: %s: %s', update, type(e).__name__, e)
    elif isinstance(e, Exception):
        log.error(
            'Exception in update %s: %s: %s', update, type(e).__name__, e, exc_info=e
        )
    else:
        log.error('Unknown error in update %s: %s', update, e)


def build_app():
    builder = (
        ApplicationBuilder()
        .token(os.environ['TELEGRAM_BOT_TOKEN'])
        .post_init(post_init)
        .post_stop(post_stop)
        .read_timeout(30)
        .write_timeout(30)
        .media_write_timeout(60)
    )

    # Looks like http://127.0.0.1:8081/
    if url := os.environ.get('LOCAL_SERVER'):
        log.info('Using local server: %s', url)
        builder = (
            builder.base_url(url + 'bot')
            .base_file_url(url + 'file/bot')
            .local_mode(True)
        )

    app = builder.build()
    app.add_error_handler(handle_error)

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

    return app


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
    build_app().run_polling()
    log.info('%s stopped.', ME)
    drop_lock()


if __name__ == '__main__':
    main()
