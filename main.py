import os
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

from util import log, log2, shorten, USER_ID, CHAN_ID, init_util
from receipt import render_receipt
from run import handle_run, handle_cmd, handle_update
from rg import handle_rg, handle_rg_callback, handle_start
from render import (
    handle_render,
    handle_doc,
    handle_render_inline_query,
    init_render,
    handle_render_callback,
    close_render,
)


def auth(
    func: Callable[[Update, ContextTypes.DEFAULT_TYPE], Coroutine],
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Coroutine]:
    @functools.wraps(func)
    async def wrapper(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if user:
            src = f'{user.full_name} ({user.name} {user.id})'
            valid = user.id == USER_ID
        else:
            chat = update.effective_chat
            if not chat:
                raise ValueError('Invalid message')
            src = f'Chat {chat.id} ({chat.type})'
            valid = chat.id == CHAN_ID

        if msg := update.message:
            log.info('%s: %s', src, shorten(msg.text))
        elif update.callback_query:
            log.info('%s: callback %s', src, shorten(update.callback_query.data))
        elif update.channel_post:
            log.info('%s: channel post %s', src, shorten(update.channel_post.text))
        elif update.edited_message:
            log.info('%s: edited %s', src, shorten(update.edited_message.text))
        elif update.edited_channel_post:
            log.info(
                '%s: edited post %s',
                src,
                shorten(update.edited_channel_post.text),
            )
        elif update.inline_query:
            log.info(
                '%s: inline query %s',
                src,
                shorten(update.inline_query.query),
            )
        else:
            log.info('%s: unknown: %s', src, update)

        if not valid:
            log2.warning('Drop message from unknown chat: %s', src)
            return

        try:
            return await func(update, ctx)
        except Exception as e:
            r = f'{type(e).__name__}: {str(e)}'
            log2.exception('%s', r)
            if msg:
                await msg.reply_text(r, do_quote=True)

    return wrapper


async def handle_msg(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        if post := update.edited_channel_post or update.channel_post:
            return await handle_doc(post)
        return

    text = update.message.text
    if not (text and text.strip()):
        with open('out.ogg', 'rb') as f:
            return await update.message.reply_voice(f, do_quote=True)
    if text.startswith('/'):
        return await handle_cmd(update, text[1:].strip())
    if '\n' not in text:
        return await handle_rg(update, ctx)
    await update.message.reply_text(render_receipt(text), do_quote=True)


async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    callback = update.callback_query
    data = callback.data
    try:
        if data:
            if data.startswith('rg_back_'):
                await handle_rg_callback(data)
            elif data[0] in ':-+':
                await handle_render_callback(update, ctx, data)
            else:
                log2.warning('bad callback: %s', data)
        else:
            log2.warning('empty callback')
    finally:
        await callback.answer()


async def handle_inline_query(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.inline_query
    log.info(
        'Inline query from %s: %s', query.from_user.full_name, shorten(query.query)
    )
    if query and (text := query.query.strip()):
        return await handle_render_inline_query(query, text)


commands = tuple(
    (f.__name__[f.__name__.index('_') + 1 :], f)
    for f in [handle_start, handle_run, handle_update, handle_rg, handle_render]
)


async def post_init(app: Application) -> None:
    bot: Bot = app.bot
    init_render('doc.db')
    init_util(bot)
    await bot.set_my_commands(tuple((s, s) for s, _ in commands))
    log2.info('Sendai initialized.')


async def post_stop(_: Application) -> None:
    close_render()
    log.info('Database closed.')


def main():
    app = (
        ApplicationBuilder()
        .token(os.environ['TELEGRAM_BOT_TOKEN'])
        .post_init(post_init)
        .post_stop(post_stop)
        .build()
    )

    for name, func in commands:
        app.add_handler(CommandHandler(name, auth(func)))

    app.add_handler(CallbackQueryHandler(auth(handle_callback)))
    app.add_handler(InlineQueryHandler(auth(handle_inline_query)))
    app.add_handler(MessageHandler(None, auth(handle_msg)))

    log.info('Starting Sendai...')
    app.run_polling()
    log.info('Sendai stopped.')


if __name__ == '__main__':
    main()
