import os
import logging
import functools
from math import floor
from typing import Hashable
from collections import defaultdict

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler

USER_ID = os.environ['USER_ID']


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
        self.kind = kind  # Total cost is floor-ed for each group by kind

    @property
    def adj(self):
        return self.cost - self.raw

    def __str__(self):
        return render_pair(self.cost, self.adj)

    __repr__ = __str__


def alter(items: list[Value], real_tot: int):
    # Alter the last value to match the total
    d = sum(item.cost for item in items) - real_tot
    n = len(items)
    if d:
        assert -n < d < n
        t = sorted(items, key=lambda x: x.adj)
        if d > 0:
            for x in t[-d:]:
                x.cost -= 1
        else:
            for x in t[:-d]:
                x.cost += 1

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
        if len(parts) == 2 and not parts[0].isdigit():
            cmd, arg = parts
            if cmd == '=':
                if the_real_tot:
                    raise SyntaxError('Unexpected "=" command')
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
            v = calc(parts[0]) * tax
            items.append(Value(v, tax))
        else:
            v = functools.reduce(lambda x, y: x * y, map(calc, parts))
            items.append(Value(v, parts[1]))

    if not items:
        raise ValueError('No values provided')

    tot = sum(item.raw for item in items)

    tot_dict = defaultdict(float)
    for item in items:
        tot_dict[item.kind] += item.raw
    real_tot = sum(floor(v) for v in tot_dict.values())

    if the_real_tot is not None and the_real_tot != real_tot:
        raise ValueError(f'{the_real_tot} != {real_tot}')

    alter(items, real_tot)
    results = [render_pair(real_tot, real_tot - tot)]

    dss = [items]
    for alt_tot, base in zip(alt_tots, alt_bases):
        alt_items = [Value(x.raw * alt_tot / tot) for x in items]
        dss.append(alt_items)
        alter(alt_items, alt_tot)
        r1 = alt_tot
        r2 = sum(x.raw for x in alt_items)
        if base != 1:
            r1 /= base
            r2 /= base
            for x in alt_items:
                x.cost /= base
                x.raw /= base
        results.append(render_pair(r1, r1 - r2))

    res = [' | '.join(str(dss[j][i]) for j in range(len(dss))) for i in range(len(items))]
    res.append('= ' + ' | '.join(results))
    return '\n'.join(res)


async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.text or not msg.from_user or not update.effective_chat:
        raise ValueError('Invalid message')

    text = msg.text.strip()
    rendered_text = text.replace('\n', ' ').replace('\r', ' ')
    if len(rendered_text) > 30:
        rendered_text = rendered_text[:27]
    user = msg.from_user

    log.info('%s (%s %s): %s', user.full_name, user.name, user.id, rendered_text)
    if user.id != USER_ID:
        log.info('Drop message from unknown user')
        return

    try:
        r = _handle(text)
    except Exception as e:
        r = f'{type(e).__name__}: {str(e)}'
        log.exception(r)

    await context.bot.send_message(chat_id=update.effective_chat.id, text=r)


def main():
    token = os.environ['TELEGRAM_BOT_TOKEN']
    application = ApplicationBuilder().token(token).build()

    application.add_handler(MessageHandler(None, handle))
    log.info('Starting Inoue bot...')
    application.run_polling()
    log.info('Inoue Bot stopped.')

if __name__ == '__main__':
    main()
