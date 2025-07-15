import os
import logging
import functools
from math import floor
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler

USER_ID = 545183649


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


def render_val(v: tuple[int, float | int] | int) -> str:
    if isinstance(v, tuple):
        if v[1] == 0:
            v = v[0]
        else:
            return f'{v[0]} ({render_sign(v[1])})'

    return str(v)


def _handle(text: str):
    real_tot: int | None = None
    tot: float = 0
    raw_vals: list[float] = []
    vals: list[int] = []
    tax: float = 1.08

    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue

        part = parts[0]

        if len(parts) == 2 and not part.isdigit():
            cmd = part
            part = parts[1]
            if cmd == '=':
                if real_tot:
                    raise SyntaxError('Unexpected "=" command')
                real_tot = int(part)
            elif cmd.startswith('t'):
                tax = calc(part)
            else:
                raise SyntaxError(f'Unknown command: {cmd}')
            continue

        if len(parts) == 1:
            v = calc(part) * tax
        else:
            v = functools.reduce(lambda x, y: x * y, map(calc, parts))

        tot += v
        raw_vals.append(v)
        vals.append(round(v))

    if not vals:
        raise ValueError('No values provided')

    if not real_tot:
        real_tot = floor(tot)

    # Alter the last value to match the total
    d = sum(vals) - real_tot
    n = len(vals)
    pairs = []
    for i in range(n):
        pairs.append((vals[i], vals[i] - raw_vals[i]))

    if d:
        assert d <= n
        for i in sorted(range(n), key=lambda i: vals[i] - raw_vals[i])[-d:]:
            p = pairs[i]
            pairs[i] = (p[0] - 1, p[1] - 1)

    assert real_tot == sum(v[0] if isinstance(v, tuple) else v for v in pairs)

    d = tot - real_tot
    pairs.append((f'= {real_tot}', -d))
    return '\n'.join(map(render_val, pairs))


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
