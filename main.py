import os
import logging
import functools
from math import floor
from typing import Callable, Coroutine, Hashable
from collections import defaultdict

from telegram import Update, Bot
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    Application,
)

from run import *
from rg import handle_rg, handle_rg_callback, handle_start
from render import handle_render, handle_doc, init_render, handle_render_callback

USER_ID = int(os.environ['USER_ID'])
CHAN_ID = int(os.environ['CHAN_ID'])


def get_logger(name):
    if os.environ.get('DEBUG') == '1':
        level = logging.DEBUG
    else:
        level = logging.INFO

    if 'JOURNAL_STREAM' in os.environ:
        fmt = '[%(levelname)s] %(message)s'
    else:
        fmt = '%(asctime)s [%(levelname)s] %(message)s'

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    h = logging.StreamHandler()
    h.setLevel(level)
    h.setFormatter(logging.Formatter(fmt))
    logger.addHandler(h)
    return logger


log = get_logger('inoue')


def calc(s: str) -> float:
    if s == 'x':
        return 1.1
    return eval(s)


def render_sign(v: float | int) -> str:
    if isinstance(v, int):
        return f'{v:+}'
    return f'{v:+.3}'


def render_pair(x, y) -> str:
    if y == 0:
        return str(x)
    return f'{x} ({render_sign(y)})'


class Value:
    def __init__(self, raw: float, kind: Hashable = None):
        self.raw = raw
        self.cost = round(raw)
        self.kind = kind  # Total cost is rounded for each group by kind

    @property
    def adj(self):
        return self.cost - self.raw

    def __str__(self):
        return render_pair(self.cost, self.adj)

    __repr__ = __str__


def alter(items: list[Value], real_tot: int):
    d = sum(item.cost for item in items) - real_tot
    if not d:
        return

    t = sorted(items, key=lambda x: x.adj, reverse=d > 0)
    off = -1 if d > 0 else 1
    d = abs(d)
    assert d <= len(t)

    for x in t[:d]:
        x.cost += off

    assert real_tot == sum(item.cost for item in items)


def _handle(text: str):
    the_real_tot: int | None = None
    tax: float = 1.08
    items: list[Value] = []
    alt_tots: list[float] = []
    alt_bases: list[int] = []

    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) == 2 and not parts[0][0].isdigit():
            cmd, arg = parts
            if cmd == '=':
                if the_real_tot is not None:
                    raise SyntaxError('Duplicate "="')
                the_real_tot = int(arg)
            elif cmd.startswith('t'):
                tax = calc(arg)
            elif cmd.startswith('a'):
                # Find the precision of the argument
                p = arg.find('.')
                if p >= 0:
                    base = 10 ** (len(arg) - p - 1)
                    alt_bases.append(base)
                    alt_tots.append(int(calc(arg) * base))
                else:
                    alt_bases.append(1)
                    alt_tots.append(calc(arg))
            else:
                raise SyntaxError(f'Unknown command: {cmd}')
            continue

        if len(parts) == 1:
            items.append(Value(calc(parts[0]) * tax, tax))
        else:
            parts = tuple(calc(p) for p in parts)
            items.append(Value(functools.reduce(lambda x, y: x * y, parts), parts[1]))

    if not items:
        raise ValueError('No values provided')

    tot_dict = defaultdict(float)
    for item in items:
        tot_dict[item.kind] += item.raw
    real_tot = sum(floor(v) for v in tot_dict.values())

    if not real_tot:
        raise ValueError('You are free!')

    if the_real_tot is not None and the_real_tot != real_tot:
        raise ValueError(f'{the_real_tot} != {real_tot}')

    alter(items, real_tot)
    results = [render_pair(real_tot, real_tot - sum(item.raw for item in items))]

    dss = [items]
    for tot, base in zip(alt_tots, alt_bases):
        alt_items = [Value(x.cost * tot / real_tot) for x in items]
        alter(alt_items, tot)
        if base != 1:
            tot /= base
            for x in alt_items:
                x.cost /= base
                x.raw /= base
        results.append(f'{tot} (x{tot / real_tot:.6f})')
        dss.append(alt_items)

    res = [
        ' | '.join(str(dss[j][i]) for j in range(len(dss))) for i in range(len(items))
    ]
    res.append('= ' + ' | '.join(results))
    return '\n'.join(res)


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

        def shorten(s: str | None) -> str:
            if s is None:
                return 'None'
            s = s.strip().replace('\n', ' ').replace('\r', ' ')
            if len(s) > 30:
                return s[:30] + '...'
            return s

        if msg := update.message:
            if msg.text:
                log.info('msg: %s', msg)

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
        else:
            log.info('%s: unknown: %s', src, update)

        if not valid:
            log.info('Drop message from unknown chat')
            return

        try:
            return await func(update, ctx)
        except Exception as e:
            r = f'{type(e).__name__}: {str(e)}'
            log.exception('%s', r)
            if msg:
                await msg.reply_text(r, do_quote=True)

    return wrapper


@auth
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
    await update.message.reply_text(_handle(text), do_quote=True)


async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    callback = update.callback_query
    await callback.answer()

    data = callback.data
    if data.startswith('rg_back_'):
        await handle_rg_callback(data)
    elif data.startswith(':'):
        await handle_render_callback(update, ctx, data)


async def post_init(app: Application) -> None:
    bot: Bot = app.bot
    await bot.set_my_commands(
        (
            ('render', 'render'),
            ('start', 'start'),
            ('run', 'run'),
            ('update', 'update'),
            ('rg', 'rg'),
        )
    )
    await bot.send_message(USER_ID, 'Inoue bot started.')
    init_render('doc.db')


def main():
    token = os.environ['TELEGRAM_BOT_TOKEN']
    app = ApplicationBuilder().token(token).post_init(post_init).build()

    app.add_handler(CommandHandler('start', auth(handle_start)))
    app.add_handler(CommandHandler('run', auth(handle_run)))
    app.add_handler(CommandHandler('update', auth(handle_update)))
    app.add_handler(CommandHandler('rg', auth(handle_rg)))
    app.add_handler(CommandHandler('render', auth(handle_render)))

    app.add_handler(CallbackQueryHandler(auth(handle_callback)))
    app.add_handler(MessageHandler(None, handle_msg))

    log.info('Starting Inoue bot...')
    app.run_polling()
    log.info('Inoue Bot stopped.')


if __name__ == '__main__':
    main()
