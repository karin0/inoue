import os
import re
import asyncio
import logging
import traceback

from html import escape as html_escape
from typing import Awaitable, Callable, Sequence, Concatenate
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar

from telegram import Message, Bot, MessageEntity, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, MessageLimit
from telegram.error import BadRequest

from db import db
from context import *


def list_env(key: str, sep: str = ',') -> tuple[str, ...]:
    if val := os.environ.get(key):
        return tuple(r for s in val.split(sep) if (r := s.strip()))
    return ()


def load_ids(key: str) -> tuple[int, ...]:
    return tuple(int(x) for x in list_env(key))


USER_ID = int(os.environ['USER_ID'])
CHAN_ID = int(os.environ['CHAN_ID'])
GROUP_ID = int(os.environ['GROUP_ID'])

GUEST_USER_IDS = frozenset(load_ids('GUEST_USER_IDS'))
IGNORE_CHAT_IDS = frozenset(load_ids('IGNORE_CHAT_IDS'))

TRUSTED_IDS = frozenset((USER_ID, CHAN_ID, GROUP_ID, *load_ids('TRUSTED_IDS')))

LOG_THREAD_ID = int(os.environ.get('LOG_THREAD_ID', 0)) or None

DB_FILE = os.environ.get('DB_FILE', ME_LOWER + '.db')


MAX_TEXT_LENGTH = MessageLimit.MAX_TEXT_LENGTH

msg: ContextVar[Message | None] = ContextVar('msg')
text_override: ContextVar[str | None] = ContextVar('text_override', default=None)
bot: Bot | None = None


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
        m = msg.get(None)
        asyncio.create_task(do_notify(text, message=m, revocable=self._revocable))

    @contextmanager
    def revocable(self):
        old = self._revocable
        self._revocable = True
        try:
            yield
        finally:
            self._revocable = old

    @contextmanager
    def suppress(self):
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

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    h = logging.StreamHandler()
    h.setLevel(level)
    h.setFormatter(logging.Formatter(fmt))
    logger.addHandler(h)

    logger.addHandler(notify)

    if level == logging.DEBUG:
        rc_log = logging.getLogger('render_core')
        rc_log.setLevel(logging.DEBUG)
        rc_log.addHandler(logging.FileHandler('render_core.log'))

    return logger


log = _get_logger(ME_LOWER)
is_debug = log.isEnabledFor(logging.DEBUG)
trace = log.debug if os.environ.get('TRACE') == '1' else lambda *_: None


def init_util(b: Bot):
    global bot
    bot = b


@contextmanager
def use_msg(m: Message | None, sender: Sender | None):
    # Only messages from USER_ID are allowed to be set in the context, since it's
    # used for `do_notify` to notify system events.
    if m and m.chat_id != USER_ID:
        m = None

    token = msg.set(m)
    if sender is not None:
        token_sender = current_sender.set(sender)
    try:
        yield m
    finally:
        msg.reset(token)
        if sender is not None:
            current_sender.reset(token_sender)  # type: ignore


@contextmanager
def use_text_override(text: str):
    token = text_override.set(text)
    try:
        yield
    finally:
        text_override.reset(token)


NOTIFY_LIMIT_INTERVAL_SEC = 20
NOTIFY_LIMIT_BURST = 5

notify_moments = deque(maxlen=NOTIFY_LIMIT_BURST)
notify_buf = []


async def _flush_notify_buf():
    if n := len(notify_buf):
        text = '\n'.join(notify_buf)
        text = truncate_text(text)
        notify_buf.clear()
        log.info('flushing %s buffered notifications (%s chars)', n, len(text))
        await do_notify(text)


def flush_notify_buf():
    if notify_buf:
        asyncio.create_task(_flush_notify_buf())


async def do_notify(
    text: str,
    parse_mode: str | None = None,
    *,
    message: Message | None = None,
    revocable: bool = False,
    quiet: bool = False,
    **kwargs,
):
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

    if m := message or msg.get(None):
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


type MessageSource = Message | Update | None


def get_msg(update: MessageSource) -> Message:
    if isinstance(update, Update):
        # Unlike update.effective_message, channel posts and callback queries
        # are ignored here.
        if m := update.message or update.edited_message:
            return m
    elif update is None:
        if m := msg.get():
            return m
    else:
        return update

    raise ValueError('No message')


def get_msg_arg(update: MessageSource) -> tuple[Message, str]:
    m = get_msg(update)
    s = text_override.get() or m.text or m.caption or ''

    if not s.startswith('/'):
        return m, s.strip()
    try:
        return m, s[s.index(' ') + 1 :].strip()
    except ValueError:
        return m, ''


def get_msg_url(msg_id, chat_id=None) -> str:
    if chat_id is None:
        chat_id = CHAN_ID
    chat_id = str(chat_id).removeprefix('-100')
    return f'https://t.me/c/{chat_id}/{msg_id}'


def get_deep_link_url(arg: str) -> str:
    assert bot is not None
    return f'https://t.me/{bot.username}?start={arg}'


def encode_chat_id(m: Message, default: str = 'u') -> str:
    chat_id = m.chat_id
    if chat_id == USER_ID:
        return default
    if chat_id == CHAN_ID:
        return 'c'
    if chat_id == GROUP_ID:
        return 'g'
    return f'G{chat_id}'


async def try_send_text[**P, R](
    func: Callable[Concatenate[str, P], Awaitable[R]],
    text: str,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    '''`parse_mode` must be specified via `kwargs` to enable the fallback.'''
    if not ('parse_mode' in kwargs or 'entities' in kwargs):
        return await func(text, *args, **kwargs)

    try:
        return await func(text, *args, **kwargs)
    except BadRequest as e:
        if "Can't parse entities:" in (e_str := str(e)):
            log.warning(
                'try_send_text: falling back without entities: %s %s %s %s',
                e,
                func,
                args,
                kwargs,
            )
            new_text = truncate_text(f'{e_str}\n{text}')
            kwargs.pop('parse_mode', None)
            kwargs.pop('entities', None)
            return await func(new_text, *args, **kwargs)
        raise


# Note: the return value could be `None` if `allow_not_modified` is set.
async def reply_text(
    update: MessageSource,
    text: str,
    parse_mode: str | None = None,
    reply_markup: InlineKeyboardMarkup | None = None,
    *,
    entities: Sequence[MessageEntity] | None = None,
    allow_not_modified: bool = False,
) -> Message | None:
    m = get_msg(update)
    chat_kind = encode_chat_id(m)
    if chat_kind == 'c':
        log.warning('reply_text: channel chat: %s', m)
    key = f'{chat_kind}-{m.message_id}'

    # Can only be called when `key` is missing in the cache.
    async def _do_reply_text():
        resp = await try_send_text(
            m.reply_text,
            text,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            entities=entities,
            do_quote=True,
        )

        val = str(resp.message_id)
        db[key] = val
        log.debug('Sending new response: %s -> %s', key, val)
        return resp

    if not (resp_msg_id := db.get(key)):
        return await _do_reply_text()

    resp_msg_id = int(resp_msg_id)
    log.debug('Editing cached response: %s -> %s', key, resp_msg_id)

    assert bot is not None
    try:
        resp = await try_send_text(
            bot.edit_message_text,
            text,
            m.chat.id,
            resp_msg_id,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            entities=entities,
        )
    except Exception as e:
        # Cache expired, remove it first for other coroutines.
        # We don't bypass 'Message is not modified' here, as the user side cannot
        # distinguish whether the message is being updated.
        # This behavior can be overridden by `allow_not_modified`.
        e = str(e)
        if 'Message is not modified' not in e:
            del db[key]
            fmt = 'Failed to edit response: %s -> %s: %s: %s'
            log.warning(fmt, key, resp_msg_id, type(e).__name__, e)
        elif allow_not_modified:
            log.info('Message not modified: %s -> %s', key, resp_msg_id)
            return None
        else:
            del db[key]

        return await _do_reply_text()

    assert isinstance(resp, Message)
    return resp


def shorten(s: str | None, limit: int = 30) -> str:
    if s is None:
        return 'None'
    s = s.strip().replace('\n', ' ').replace('\r', ' ')
    if len(s) > limit:
        return s[:limit] + '...'
    return s


def truncate_text(s: str) -> str:
    s = s.strip()
    if len(s) > MAX_TEXT_LENGTH:
        s = s[: MAX_TEXT_LENGTH - 12] + '\n[truncated]'
    return s


def _create_escape_trans(chars: str) -> dict[int, str]:
    return {ord(c): '\\' + c for c in chars}


# `telegram.helpers.escape_markdown` implements this with `re.sub`, which might
# be slower than us.
_MD2_TRANS = _create_escape_trans(r'\_*[]()~`>#+-=|{}.!')


def escape(s: str) -> str:
    return s.translate(_MD2_TRANS)


def escape_pre(s: str) -> str:
    # `replace` is faster for small charset.
    return s.replace('\\', '\\\\').replace('`', '\\`')


type Content = tuple[str, str | None]


# This truncates the text by default, trimming any leading and trailing spaces.
def pre_block(s: str, *, do_truncate: bool = True) -> Content:
    if len(s) <= MAX_TEXT_LENGTH:
        return pre_block_raw(s), 'MarkdownV2'

    if do_truncate:
        s = truncate_text(s)
    return s, None


def pre_block_raw(s: str, *, lang: str = '') -> str:
    return f'```{lang}\n{escape_pre(s)}\n```'


reg_cleanup = re.compile(r'\n{3,}')
reg_cleanup_pre = re.compile(r'^```\n+(.+?)\n+```$', re.DOTALL | re.MULTILINE)


def repl_cleanup_pre(m: re.Match) -> str:
    return '```\n' + m.group(1).strip() + '\n```'


def cleanup_text(s: str) -> str:
    s = reg_cleanup.sub('\n\n', s)
    s = reg_cleanup_pre.sub(repl_cleanup_pre, s)
    return s


async def _keep_action(msg: Message, action: ChatAction):
    try:
        while True:
            await msg.reply_chat_action(action)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass


@contextmanager
def keep_chat_action(msg: Message, action: ChatAction):
    '''Re-send chat action every 4s so it stays visible during long operations.'''
    task = asyncio.create_task(_keep_action(msg, action))
    try:
        yield task
    finally:
        task.cancel()
