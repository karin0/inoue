import os
import asyncio
import logging
import traceback

from collections import deque
from contextlib import contextmanager

from telegram import Message

from .ctx import get_ctx_msg
from .text import truncate_text
from .env import ME_LOWER, USER_ID, GROUP_ID, LOG_THREAD_ID


class NotifyHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self._revocable = False
        self._suppressed = False

    def emit(self, record: logging.LogRecord) -> None:
        if self._suppressed:
            return
        text = truncate_text(self.format(record))
        # Fetch the context before yielding to async code
        asyncio.create_task(
            do_notify(text, message=get_ctx_msg(), revocable=self._revocable)
        )

    @contextmanager
    def revocable(self):
        '''Note: this does not use ContextVar. Do not hold this across await.'''
        old = self._revocable
        self._revocable = True
        try:
            yield
        finally:
            self._revocable = old

    @contextmanager
    def suppress(self):
        '''Note: this does not use ContextVar. Do not hold this across await.'''
        old = self._suppressed
        self._suppressed = True
        try:
            yield
        finally:
            self._suppressed = old


notify = NotifyHandler()


def _get_logger(name):
    if os.environ.get('DEBUG') == '1':
        level = logging.DEBUG
    else:
        level = logging.INFO

    if 'JOURNAL_STREAM' in os.environ:
        fmt = '[%(levelname)s] %(message)s'
    else:
        fmt = '%(asctime)s [%(levelname)s] %(message)s'

    fmt = logging.Formatter(fmt)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    h = logging.StreamHandler()
    h.setLevel(level)
    h.setFormatter(fmt)
    logger.addHandler(h)

    logger.addHandler(notify)

    if level == logging.DEBUG:
        rc_log = logging.getLogger('render_core')
        rc_log.setLevel(logging.DEBUG)
        rc_log.propagate = False
        rc_log.handlers.clear()

        h = logging.FileHandler('render_core.log')
        h.setLevel(logging.DEBUG)
        h.setFormatter(fmt)
        rc_log.addHandler(h)

        h = logging.StreamHandler()
        h.setLevel(logging.INFO)
        h.setFormatter(fmt)
        rc_log.addHandler(h)

    return logger


log = _get_logger(ME_LOWER)
is_debug = log.isEnabledFor(logging.DEBUG)
trace = log.debug if os.environ.get('TRACE') == '1' else lambda *_: None

NOTIFY_LIMIT_INTERVAL_SEC = 20
NOTIFY_LIMIT_BURST = 5

notify_moments = deque(maxlen=NOTIFY_LIMIT_BURST)
notify_buf = []


def flush_notify_buf():
    if n := len(notify_buf):
        text = '\n'.join(notify_buf)
        text = truncate_text(text)
        notify_buf.clear()
        log.info('flushing %s buffered notifications (%s chars)', n, len(text))
        asyncio.create_task(do_notify(text))


async def do_notify(
    text: str,
    parse_mode: str | None = None,
    *,
    message: Message | None = None,
    revocable: bool = False,
    quiet: bool = False,
    **kwargs,
):
    from .bot import reply_text
    from . import _bot_util

    bot = _bot_util.bot

    loop = asyncio.get_event_loop()
    now = loop.time()
    while notify_moments and now - notify_moments[0] >= NOTIFY_LIMIT_INTERVAL_SEC:
        notify_moments.popleft()

    if len(notify_moments) >= NOTIFY_LIMIT_BURST:
        notify_buf.append(text)
        dt = NOTIFY_LIMIT_INTERVAL_SEC - (now - notify_moments[0]) + 1
        with notify.suppress():
            log.warning('do_notify: rate limited, flushing in %.3f secs', dt)
        loop.call_later(dt, flush_notify_buf)
        return
    notify_moments.append(now)

    if m := message or get_ctx_msg():
        try:
            if revocable:
                return await reply_text(m, text, parse_mode, **kwargs)
            else:
                return await m.reply_text(text, parse_mode, do_quote=True, **kwargs)
        except Exception as e:
            traceback.print_exc()
            text += f'\nreply_text: {type(e).__name__}: {e}'
            text = truncate_text(text)

    if bot:
        try:
            if quiet:
                await bot.send_message(
                    GROUP_ID, text, parse_mode, disable_notification=True, **kwargs
                )
            else:
                await bot.send_message(
                    USER_ID, text, parse_mode, message_thread_id=LOG_THREAD_ID, **kwargs
                )
        except Exception:
            with notify.suppress():
                log.exception('do_notify: send_message failed')
