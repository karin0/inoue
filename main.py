import os
import asyncio
import logging
import functools
from typing import Callable, Coroutine, Iterable

from telegram import Message, User, Update, Bot, MessageOriginChannel
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
from telegram.constants import ChatID

from util import (
    log,
    notify,
    pre_block,
    reply_text,
    shorten,
    USER_ID,
    CHAN_ID,
    GROUP_ID,
    GUEST_USER_IDS,
    DB_FILE,
    ME,
    init_util,
    use_msg,
    use_is_guest,
    get_msg,
    get_msg_arg,
    do_notify,
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
    handle_render_group,
    handle_render_inline_query,
    ALL_CALLBACK_SIGNS,
)
from db import db


def auth(
    func: Callable[[Update, ContextTypes.DEFAULT_TYPE], Coroutine],
    *,
    permissive: bool = False,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Coroutine]:
    @functools.wraps(func)
    async def wrapper(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        effective_msg = update.effective_message
        src = None
        sender_id = None
        sender_name = None
        valid = False

        if sender := update.effective_sender:
            sender_id = sender.id
            if isinstance(sender, User):
                sender_name = sender.full_name
                src = f'{sender_name} ({sender.name} {sender_id})'
                valid = sender_id == USER_ID
            else:  # Chat
                sender_name = sender.title
                src = f'{sender_name} [{sender.type} {sender_id}]'

        if chat := update.effective_chat:
            if chat.id != sender_id:
                src2 = f'{chat.title} {{{chat.type} {chat.id}}}'
                src = f'{src} @ {src2}' if src else src2

            if not valid and permissive:
                # Only `handle_msg` accepts messages that are not from USER_ID.
                valid = chat.id == CHAN_ID or (
                    # The content must be from CHAN_ID to be trusted, even if
                    # auto-forwarded to GROUP_ID.
                    chat.id == GROUP_ID
                    and (msg := effective_msg)
                    and (from_user := msg.from_user)
                    and from_user.id == ChatID.SERVICE_CHAT
                    and msg.is_automatic_forward
                    and isinstance(origin := msg.forward_origin, MessageOriginChannel)
                    and origin.chat.id == CHAN_ID
                )

        if is_guest := not valid and sender_id in GUEST_USER_IDS:
            src = f'{src} (guest)'
            valid = permissive

        log.debug('Entering %s from %s: %s', func.__name__, src, update)

        item = callback = query = None
        if msg := update.message:
            log.info('%s: msg %s', src, shorten(msg.text))
        elif msg := update.edited_message:
            log.info('%s: edited %s', src, shorten(msg.text))
        elif item := update.channel_post:
            log.info('%s: channel post %s', src, shorten(item.text))
        elif item := update.edited_channel_post:
            log.info('%s: edited post %s', src, shorten(item.text))
        elif callback := update.callback_query:
            if isinstance(callback.message, Message):
                msg = callback.message
            log.info('%s: callback %s', src, shorten(callback.data))
        elif query := update.inline_query:
            log.info('%s: inline %s', src, shorten(query.query))
        else:
            log.info('%s: unknown: %s', src, update)

        if (item := msg or item) != effective_msg:
            log.warning('Message mismatch: %s vs %s', item, effective_msg)

        if not valid:
            log.warning(
                '%s: Drop unauthorized update from %s: %s\nSender: %s\nChat: %s',
                func.__name__,
                src,
                update,
                sender,
                chat,
            )
            if is_guest and msg:
                await reply_text(
                    msg,
                    f'Hello, `{escape(sender_name)}`\\! Send `/render <your template>` to begin\\.',
                    parse_mode='MarkdownV2',
                )
            return

        if is_guest:
            if msg and func == handle_msg:
                log.warning(
                    '%s said: %s\n  %s',
                    src,
                    shorten(msg.text or msg.caption or '', limit=3000),
                    update,
                )
                return

            log.warning('Allowing guest access: %s: %s', sender_id, update)

        with use_msg(msg), use_is_guest(is_guest):
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

    # `render` handles Doc messages that are forwarded from CHAN_ID to its discussion group.
    if (
        isinstance(origin := msg.forward_origin, MessageOriginChannel)
        and msg.is_automatic_forward
        and msg.chat_id == GROUP_ID
        and origin.chat.id == CHAN_ID
    ):
        return await handle_render_group(msg, origin.message_id)

    # Privileged operations are only allowed in private chats, even if it's from
    # USER_ID.
    if msg.chat_id != USER_ID:
        log.warning('handle_msg: unauthorized update: %s', update)
        return

    # ID Bot
    if origin:
        return await msg.reply_text(*pre_block(str(origin)), do_quote=True)

    if not ((text := msg.text) and text.strip()):
        with open('out.ogg', 'rb') as f:
            return await msg.reply_voice(f, do_quote=True)

    if text.startswith('/'):
        return await handle_cmd(msg, text[1:].strip())

    if '\n' not in text:
        return await handle_rg(update, ctx)

    await reply_text(msg, *pre_block(render_receipt(text)))


async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    callback = update.callback_query
    data = callback.data
    fut = None
    if data:
        if data.startswith('rg_'):
            fut = handle_rg_callback(data)
        elif data[0] in ALL_CALLBACK_SIGNS:
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


def stats(header=ME) -> tuple[str]:
    info = f'{header}: {db.summary()}'
    log.info('%s', info)
    text = f'{escape(greeting())}\n```\n{escape(info)}\n```'
    return text, 'MarkdownV2'


async def handle_greet(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await reply_text(update, *stats())


async def handle_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg, arg = get_msg_arg(update)
    if arg and arg.startswith('rg_'):
        return await handle_rg_start(msg, arg)

    return await reply_text(msg, *stats())


def parse_sort(arg: str) -> Iterable[int]:
    last = None
    for x in arg.split():
        if last and len(x) < len(last):
            # last = 10086, x = 89 -> out = 10089
            x = last[: len(last) - len(x)] + x
        last = x
        yield int(x)


async def handle_sort(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg, arg = get_msg_arg(update)
    if not arg:
        return await msg.reply_text('Usage: /sort 114 514 1919 810 ...')
    res = '\n'.join(str(x) for x in sorted(parse_sort(arg)))
    await reply_text(msg, *pre_block(res))


commands = tuple(
    (f.__name__[f.__name__.index('_') + 1 :], f)
    for f in [
        handle_start,
        handle_greet,
        handle_run,
        handle_update,
        handle_rg,
        handle_render,
        handle_sort,
    ]
)


async def post_init(app: Application) -> None:
    bot: Bot = app.bot
    init_util(bot)
    db.connect(DB_FILE)
    await bot.set_my_commands(tuple((s, s) for s, _ in commands))
    if not log.isEnabledFor(logging.DEBUG):
        await do_notify(*stats(f'{ME} initiated'))


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
            f = rg = auth(func)
        elif name == 'render':
            f = auth(func, permissive=True)
        else:
            f = auth(func)
        app.add_handler(CommandHandler(name, f))

    for off in range(5):
        app.add_handler(CommandHandler(f'rg{off}', rg))

    app.add_handler(CallbackQueryHandler(auth(handle_callback, permissive=True)))
    app.add_handler(InlineQueryHandler(auth(handle_inline_query, permissive=True)))
    app.add_handler(MessageHandler(None, auth(handle_msg, permissive=True)))

    log.info('Starting %s...', ME)
    app.run_polling()
    log.info('%s stopped.', ME)


if __name__ == '__main__':
    main()
