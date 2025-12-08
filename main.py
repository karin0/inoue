import os
import asyncio
import functools
from typing import Callable, Coroutine

from telegram import Update, Bot
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    InlineQueryHandler,
    Application,
)
from telegram.error import NetworkError

from util import (
    log,
    pre_block,
    reply_text,
    shorten,
    USER_ID,
    CHAN_ID,
    GROUP_ID,
    init_util,
    use_msg,
    get_msg,
    get_msg_arg,
    do_notify,
    notify,
    escape,
)
from motto import greeting
from inoue import render_receipt
from run import handle_run, handle_cmd, handle_update
from rg import handle_rg, handle_rg_callback, handle_rg_start
from render import (
    handle_render,
    handle_render_doc,
    handle_render_callback,
    handle_render_group_msg,
    handle_render_inline_query,
    CALLBACK_SIGNS,
)
from db import db


def auth(
    func: Callable[[Update, ContextTypes.DEFAULT_TYPE], Coroutine],
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Coroutine]:
    @functools.wraps(func)
    async def wrapper(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        chat = update.effective_chat
        src = 'Unknown source'
        valid = False

        if user := update.effective_user:
            src = f'{user.full_name} ({user.name} {user.id})'
            valid = user.id == USER_ID

        if not valid and chat:
            src = f'{chat.title} [{chat.type} {chat.id}]'
            valid = chat.id == CHAN_ID or chat.id == GROUP_ID

        log.debug('Entering %s from %s: %s', func.__name__, src, update)

        if msg := update.message:
            log.info('%s: msg %s', src, shorten(msg.text))
        elif msg := update.edited_message:
            log.info('%s: edited %s', src, shorten(msg.text))
        elif item := update.channel_post:
            log.info('%s: channel post %s', src, shorten(item.text))
        elif item := update.edited_channel_post:
            log.info('%s: edited post %s', src, shorten(item.text))
        elif item := update.callback_query:
            msg = update.callback_query.message
            log.info('%s: callback %s', src, shorten(item.data))
        elif item := update.inline_query:
            log.info('%s: inline %s', src, shorten(item.query))
        else:
            log.info('%s: unknown: %s', src, update)

        item = msg or item
        if item != update.effective_message:
            log.warning('Message mismatch: %s vs %s', item, update.effective_message)

        if not valid:
            log.warning('Drop unauthorized update from %s: %s', src, update)
            return

        with use_msg(msg):
            try:
                return await func(update, ctx)
            except Exception as e:
                with notify.revocable():
                    # Can be edited to successful responses later after user edits
                    log.exception('%s: %s: %s', func.__name__, type(e).__name__, e)

    return wrapper


async def handle_msg(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if post := update.edited_channel_post or update.channel_post:
        return await handle_render_doc(post)

    if not (msg := get_msg(update)):
        return

    if (chat := update.effective_chat) and chat.id == GROUP_ID:
        return await handle_render_group_msg(msg)

    if src := msg.forward_origin:
        # ID Bot
        return await msg.reply_text(**pre_block(str(src)))

    text = msg.text
    if not (text and text.strip()):
        with open('out.ogg', 'rb') as f:
            return await msg.reply_voice(f, do_quote=True)

    if text.startswith('/'):
        return await handle_cmd(update, text[1:].strip())

    if '\n' not in text:
        return await handle_rg(update, ctx)

    await reply_text(update, **pre_block(render_receipt(text)))


async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    callback = update.callback_query
    data = callback.data
    fut = None
    if data:
        if data.startswith('rg_'):
            fut = handle_rg_callback(data)
        elif data[0] in CALLBACK_SIGNS:
            fut = handle_render_callback(update, ctx, data)
        elif data != 'noop':
            log.warning('bad callback: %s', data)
    else:
        log.warning('empty callback')

    if fut:
        try:
            await asyncio.gather(fut, callback.answer())
        except Exception:
            await callback.answer()
            raise
    else:
        await callback.answer()


async def handle_inline_query(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.inline_query
    data = query.query.strip()
    if query and data:
        await handle_render_inline_query(query, data)


def stats() -> dict[str]:
    s = db.summary()
    text = f'{escape(greeting())}\n```\nSendai initiated: {escape(s)}\n```'
    return {'text': text, 'parse_mode': 'MarkdownV2', 'do_quote': True}


async def handle_greet(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await get_msg(update).reply_text(**stats())


async def handle_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg, arg = get_msg_arg(update)
    if arg and arg.startswith('rg_'):
        return await handle_rg_start(msg, arg)

    return await msg.reply_text(**stats())


commands = tuple(
    (f.__name__[f.__name__.index('_') + 1 :], f)
    for f in [
        handle_start,
        handle_greet,
        handle_run,
        handle_update,
        handle_rg,
        handle_render,
    ]
)


async def post_init(app: Application) -> None:
    bot: Bot = app.bot
    init_util(bot)
    db.connect('sendai.db')
    await bot.set_my_commands(tuple((s, s) for s, _ in commands))
    await do_notify(**stats())
    log.info('Sendai initiated: %s', db.summary())


async def post_stop(_: Application) -> None:
    db.close()
    log.info('Database closed.')


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


def main():
    app = (
        ApplicationBuilder()
        .token(os.environ['TELEGRAM_BOT_TOKEN'])
        .post_init(post_init)
        .post_stop(post_stop)
        .build()
    )
    app.add_error_handler(handle_error)

    for name, func in commands:
        if name == 'rg':
            f = rg = auth(handle_rg)
        else:
            f = auth(func)
        app.add_handler(CommandHandler(name, f))

    for off in range(5):
        app.add_handler(CommandHandler(f'rg{off}', rg))

    app.add_handler(CallbackQueryHandler(auth(handle_callback)))
    app.add_handler(InlineQueryHandler(auth(handle_inline_query)))
    app.add_handler(MessageHandler(None, auth(handle_msg)))

    log.info('Starting Sendai...')
    app.run_polling()
    log.info('Sendai stopped.')


if __name__ == '__main__':
    main()
