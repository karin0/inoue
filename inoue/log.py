import asyncio
import functools
import logging
import os

from collections import deque
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, override

from bot import Responder, bot, create_task, escape, truncate_text
from bot import log as bot_log

from .ctx import get_ctx_rs
from .env import GROUP_ID, LOG_THREAD_ID, MAX_TEXT_LENGTH, ME_LOWER, USER_ID
from .text import escape_pre


class NotifyHandler(logging.Handler):
    __slots__ = ('_revocable', '_suppressed')

    def __init__(self):
        super().__init__(logging.WARNING)
        self.setFormatter(Formatter())
        self._revocable = False
        self._suppressed = False

    def _format2(self, record: logging.LogRecord) -> tuple[str, str | None]:
        exc = record.exc_info
        ext = record.exc_text
        record.exc_info = record.exc_text = None
        fmt = self.formatter
        assert fmt

        res = fmt.format(record).strip()
        if not ext:
            if not exc:
                return truncate_text(res), None
            ext = fmt.formatException(exc)

        record.exc_info = exc
        record.exc_text = ext

        new_ext = truncate_text(ext, limit=MAX_TEXT_LENGTH - len(res) - 1)
        res = f'{escape(res)}\n```\n{escape_pre(new_ext)}```'
        return res, 'MarkdownV2'

    def emit(self, record: logging.LogRecord) -> None:
        if self._suppressed:
            return

        # Fetch the context before yielding to async.
        create_task(do_notify(*self._format2(record), rs=get_ctx_rs(), revocable=self._revocable))

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


class Formatter(logging.Formatter):
    __slots__ = ()
    __BOT_TOKEN = os.environ.pop('TELEGRAM_BOT_TOKEN')

    @override
    def format(self, record):
        return super().format(record).replace(self.__BOT_TOKEN, '***')


class MicrosecondFormatter(Formatter):
    __slots__ = ()

    @override
    def formatTime(self, record, datefmt=None):
        return datetime.fromtimestamp(record.created).strftime('%m-%d %H:%M:%S.%f')


FILTER_WORDS = tuple(os.environ.get('LOG_FILTER_WORDS', '').split())


class Filter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if FILTER_WORDS and record.levelno <= logging.INFO:
            s = record.getMessage()
            return not any(word in s for word in FILTER_WORDS)
        return True


notify = NotifyHandler()

TRACE = os.environ.get('TRACE') == '1'
TRACE_LVL = 5

if TRACE:
    logging.addLevelName(TRACE_LVL, 'TRACE')


def _get_logger(name):
    if TRACE:
        level = TRACE_LVL
    elif os.environ.get('DEBUG') == '1':
        level = logging.DEBUG
    else:
        level = logging.INFO

    if 'JOURNAL_STREAM' in os.environ:
        fmt = '[%(levelname)s] %(message)s'
        fmt = Formatter(fmt)
    else:
        fmt = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        fmt = MicrosecondFormatter(fmt)

    logging.captureWarnings(True)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    h = logging.StreamHandler()
    h.setFormatter(fmt)
    root.addHandler(h)

    logger = logging.getLogger(name)
    logger.addHandler(notify)
    bot_log.addHandler(notify)

    if FILTER_WORDS:
        h.addFilter(Filter())

    for name in ('telegram', 'httpcore', 'httpx', 'asyncio'):
        logging.getLogger(name).setLevel(logging.WARNING)

    if TRACE:
        rc_log = logging.getLogger('render_core')
        rc_log.propagate = False
        rc_log.handlers.clear()

        h = logging.FileHandler('render_core.log', 'w', encoding='utf-8')
        h.setFormatter(fmt)
        rc_log.addHandler(h)

        h = logging.StreamHandler()
        h.setLevel(logging.INFO)
        h.setFormatter(fmt)
        rc_log.addHandler(h)

    return logger


if TYPE_CHECKING:

    class Logger(logging.Logger):
        trace = logging.Logger.debug
else:
    Logger = logging.Logger

log: Logger = _get_logger(ME_LOWER)  # type: ignore[assignment]
is_debug = log.isEnabledFor(logging.DEBUG)

if TRACE:
    log.trace = functools.partial(log.log, TRACE_LVL)
else:
    log.trace = lambda *_: None  # type: ignore[assignment]

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
        create_task(do_notify(text))


async def do_notify(
    text: str,
    parse_mode: str | None = None,
    *,
    rs: Responder | None = None,
    revocable: bool = False,
    quiet: bool = False,
):
    try:
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
            return None
        notify_moments.append(now)

        if not quiet and (rs := rs or get_ctx_rs()) is not None:
            try:
                return await rs.reply(
                    text, parse_mode, cached=revocable, disable_web_page_preview=True
                )
            except Exception as e:
                with notify.suppress():
                    log.exception('do_notify: reply failed')
                text += f'\nreply: {type(e).__name__}: {e}'
                text = truncate_text(text)
                parse_mode = None

        if quiet:
            await bot.send_message(
                GROUP_ID, text, parse_mode, disable_notification=True, disable_web_page_preview=True
            )
        else:
            await bot.send_message(
                USER_ID,
                text,
                parse_mode,
                message_thread_id=LOG_THREAD_ID,
                disable_web_page_preview=True,
            )
    except Exception:
        with notify.suppress():
            log.exception('do_notify: send_message failed')
