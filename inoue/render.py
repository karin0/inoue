import os
import re
import time
import asyncio
import weakref
import builtins
from pathlib import Path
from itertools import chain, islice
from typing import (
    Container,
    Iterable,
    Mapping,
    Callable,
    Awaitable,
    Type,
    cast,
)

from telegram import (
    CallbackQuery,
    InlineQuery,
    InlineQueryResultArticle,
    InputTextMessageContent,
    Message,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.constants import MessageLimit, ReactionEmoji, KeyboardButtonStyle

from render_core import Engine, Value, to_str
from bot import (
    bot,
    get_context,
    shorten,
    truncate_text,
    escape,
    Responder,
    EditHandle,
    PhotoPayload,
    DocumentPayload,
    MessageArg,
    CallbackData,
    callback_query,
    command,
)

from .db import db
from .log import log, do_notify
from .env import USER_ID, CHAN_ID, MAX_TEXT_LENGTH, list_env, encode_id
from .text import cleanup_text, pre_block
from .segments import (
    Segment,
    Element,
    Pre,
    Time,
    BlockQuote,
    Formatter,
    Bold,
    get_renderer,
    render_segment,
)
from .render_bridge import Bridge, LocalPath, to_segment
from .utils import get_msg_url, try_send_text_or_not_modified
from .render_context import OverriddenDict, encode_value, decode_value

# '/' is kept for compatibility, which was used for '-'.
BUTTON_SIGN = '!'
CALLBACK_SIGNS = '-+/' + BUTTON_SIGN
MEMORY_SIGN = '@'
PATH_SIGNS = ':#`'
CALLBACK_SPECIAL = frozenset(CALLBACK_SIGNS + MEMORY_SIGN + PATH_SIGNS)

SPECIAL_FLAG_ICONS = {
    '_pre': '📋',
    '_fold': '💌',
}


def is_safe_key(key: str | None) -> bool:
    return bool(key) and all(c not in CALLBACK_SPECIAL for c in key)


def is_safe_mem(mem: str | None) -> bool:
    return bool(mem) and all(c not in PATH_SIGNS for c in mem)


def encode_flags(flags: dict[str, bool] | None) -> str:
    if not flags:
        return ''
    return ''.join((CALLBACK_SIGNS[v] + k) for k, v in flags.items())


ENV_PREFIX = '_env.'
MEMORY_KEY = '_mem'
BUTTON_KEY = '_btn'
SPECIAL_KEYS = (MEMORY_KEY, BUTTON_KEY)
BUTTON_PREFIX = ENV_PREFIX + 'btn' + '.'


def get_env[T](ctx: Mapping[str, Value], key: str, default: T = None) -> Value | T:
    v = ctx.get(ENV_PREFIX + key)
    if v is None:
        v = ctx.get('_' + key)
        if v is None:
            return default
    return v


def get_env_flag[T](ctx: Mapping[str, Value], key: str, default: T = False) -> bool | T:
    v = get_env(ctx, key)
    if v is None:
        return default
    if v == '0':
        return False
    if v == '1':
        return True
    return bool(v)


def make_markup(
    path: str,
    ctx: Mapping[str, Value],
    current_state: 'MarkupState | None',
    doc_ids: dict[int, str] | None,
) -> tuple[InlineKeyboardMarkup | None, str | None]:
    # query data:
    #   ('-'|'+' <flag-key>)*
    #   ['!' <btn-key>]
    #   ['@' <memory>]
    #   <path-header: ':'|'#'|'`'> <path-body>
    # where '-' means 0, '+' means 1

    if current_state is None:
        state = current_data = None
    else:
        state, current_data = current_state

    size = len(path.encode('utf-8'))
    if state:
        size += sum(len(k.encode('utf-8')) for k in state.keys()) + len(state)

    if size > InlineKeyboardButton.MAX_CALLBACK_DATA:
        return None, None

    memory = ''
    if (val := ctx.get(MEMORY_KEY)) is not None and is_safe_mem(
        payload := encode_value(val)
    ):
        delta = len(payload.encode('utf-8')) + 1
        if size + delta <= InlineKeyboardButton.MAX_CALLBACK_DATA:
            memory = '@' + payload
            size += delta

    # `state`, `memory` and `ctx[BUTTON_KEY]` defines the current state, while
    # `flags` and `buttons` defines the potential next states.
    flags: dict[str, bool] = {}
    buttons = []
    for k, v in ctx.items():
        if len(btn_key := k.removeprefix(BUTTON_PREFIX)) != len(k):
            if (
                is_safe_key(btn_key)
                and (v and v != '0')
                and (
                    size + len(btn_key.encode('utf-8')) + 1
                    <= InlineKeyboardButton.MAX_CALLBACK_DATA
                )
            ):
                buttons.append(btn_key)
            continue

        if k.startswith(ENV_PREFIX) or k in SPECIAL_KEYS or not is_safe_key(k):
            continue

        if v == '0' or v == 0:  # This covers `False` as well.
            v = False
        elif v == '1' or v == 1:
            v = True
        else:
            continue

        if size + len(k.encode('utf-8')) + 1 <= InlineKeyboardButton.MAX_CALLBACK_DATA:
            flags[k] = v

    log.debug('make_markup: flags %s state %s memory %r', flags, state, memory)

    # The external state takes precedence even over `ctx.overrides`, but the existing
    # buttons should occur before in the order.
    if state:
        flags.update(state)

    if doc_ids and not get_env_flag(ctx, 'ref', True):
        doc_ids = None

    if current_data == path:
        current_data = None

    row: list[InlineKeyboardButton | None] = [None]

    def push_button(name: str, key: str):
        data = encode_flags(state) + BUTTON_SIGN + key + memory + path
        row.append(InlineKeyboardButton(name, callback_data=data))

    def push_flag(name: str):
        data = encode_flags(state) + memory + path
        row.append(InlineKeyboardButton(name, callback_data=data))

    for k in buttons:
        if icon := get_env(ctx, 'icon.' + k):
            label = str(icon)
        else:
            label = k
        push_button(label, k)

    if flags:
        if state is None:
            state = {}

        hide_flags = get_env_flag(ctx, 'hide_flags')
        for k, v in flags.items():
            old = k in state
            log.debug('make_markup: flag %s in=%s new=%s s=%s', k, old, v, state)
            state[k] = not v

            if (icon := get_env(ctx, 'icon.' + k)) is not None:
                if label := to_str(icon):
                    push_flag(label + (':=' if old else '=') + '01'[v])
                else:
                    log.debug('make_markup: hiding flag %s due to empty icon', k)
            elif not hide_flags:
                label = SPECIAL_FLAG_ICONS.get(k, k)
                push_flag(label + (':=' if old else '=') + '01'[v])
            else:
                log.debug('make_markup: hiding flag %s', k)

            if old:
                state[k] = v
            else:
                del state[k]

    log.debug('make_markup: final state %s (%s)', state, current_data)

    if current_data:
        row.append(
            InlineKeyboardButton(
                '🔄', callback_data=current_data, style=KeyboardButtonStyle.PRIMARY
            )
        )

    if doc_ids:
        if len(doc_ids) == 1:
            doc_id = next(iter(doc_ids))
            row.append(
                InlineKeyboardButton(
                    '🔗', get_msg_url(doc_id), style=KeyboardButtonStyle.SUCCESS
                )
            )
        else:
            for doc_id, doc_name in doc_ids.items():
                row.append(
                    InlineKeyboardButton(
                        f'🔗{doc_name}',
                        get_msg_url(doc_id),
                        style=KeyboardButtonStyle.SUCCESS,
                    )
                )

    if len(row) <= 1:
        return None, None

    row[0] = InlineKeyboardButton(
        ('⏪ ' if current_data else '🔄 ') + shorten(path[1:]),
        callback_data=path,
        style=(
            KeyboardButtonStyle.DANGER if current_data else KeyboardButtonStyle.PRIMARY
        ),
    )
    row_ = cast(list[InlineKeyboardButton], row)

    row_limit_ = get_env(ctx, 'btns_per_row', 5)
    try:
        if isinstance(row_limit_, (str, int, float)):
            row_limit = int(row_limit_)
        else:
            row_limit = 5
    except ValueError, TypeError:
        row_limit = 5

    # Group 5 buttons per row
    rows = [row_[i : i + row_limit] for i in range(0, len(row), row_limit)]
    return InlineKeyboardMarkup(rows), current_data


type MessageSpec = tuple[str, str | None, InlineKeyboardMarkup | None]
type UpdateCallback = Callable[
    [MessageSpec], Awaitable[Message | bool | EditHandle | None]
]
type MarkupState = tuple[dict[str, bool], str]

OVERFLOWED_TEXT = '…\n'


class RenderContext:
    __slots__ = (
        '_markup_state',
        '_path',
        '_update_callback',
        '_doc_refs',
        '_render_time',
        '_trusted',
        '_can_escalate',
        '_as_caption',
        '_responder',
        'data',
        'engine',
        '__weakref__',
    )

    def __init__(
        self,
        overrides: dict[str, Value] | None = None,
        markup_state: MarkupState | None = None,
        as_caption: bool = False,
        doc_id: int | None = None,
        path: str | None = None,
        update_callback: UpdateCallback | None = None,
        responder: Responder | None = None,
    ):
        self._markup_state = markup_state
        self._as_caption = as_caption
        self._responder = responder
        self.set_path(path)
        self.set_update_callback(update_callback)

        self._doc_refs = {doc_id: ''} if doc_id is not None else {}
        self._render_time = None

        if overrides is None:
            overrides = {}

        trusted = source = None

        context = get_context()
        if (sender := context.sender) is not None and sender.id in (USER_ID, CHAN_ID):
            overrides['_trusted'] = trusted = sender.id

        # Existing `overrides` are frozen and immutable in `Engine`, so this is safe.
        update = context.update
        if (user := update.effective_user) is not None:
            overrides['_user_id'] = user.id
            overrides['_user_name'] = source = user.full_name
        if (chat := update.effective_chat) is not None:
            overrides['_chat_id'] = chat.id
            if title := chat.title:
                overrides['_chat_title'] = title
                source = f'{title} @ {source}' if source else title
        if source:
            overrides['_source'] = source
        if (msg := update.effective_message) is not None:
            overrides['_msg_id'] = msg.message_id
            if (reply := msg.reply_to_message) is not None:
                overrides['_replied'] = repr(reply)

        log.info('create_engine: %s', overrides)

        # We assume contents from USER_ID and saved docs are trusted.
        # However, docs (typically in ALLOWED_GUEST_DOC_PREFIXES) need to escalate
        # explicitly if expanded from a guest context, declaring that their security
        # does not rely on any context values, which can be spoofed by guests via
        # callback data.
        self._trusted = trusted
        self._can_escalate = trusted or doc_id

        self.data = OverriddenDict({}, overrides)
        bridge = Bridge(self.data, trusted, self)

        # Access to attributes with underscores should be forbidden in `simpleeval`,
        # so `os` is safe.
        self.data['os'] = self.data['sys'] = bridge

        this = weakref.ref(self)

        def doc_loader(name: str) -> str | None:
            return this()._doc_loader(name)  # type: ignore

        self.engine = Engine(self.data, doc_loader, funcs=bridge._get_func)

    def _error(self, msg: str) -> None:
        self.engine.errors.append(msg)

    def set_path(self, path: str | None) -> None:
        self._path = path

    def set_update_callback(self, callback: UpdateCallback | None) -> None:
        self._update_callback = callback and (callback, asyncio.Lock())

    async def _invoke_update_callback(self, spec: MessageSpec):
        if self._update_callback is not None:
            func, lock = self._update_callback
            async with lock:
                return await func(spec)

    def _escalate(self) -> int | None:
        if (token := self._can_escalate) is not None:
            log.info('Escalating privileges: %r %r', self._trusted, token)
            self._trusted = token
            self.data.overrides['_trusted'] = token
            return token

        log.warning('Escalation rejected: %r', self._trusted)

    def _doc_loader(self, name: str) -> str | None:
        row = get_doc(name, bool(self._trusted))
        if row is None:
            return None
        if (doc_id := row[0]) is not None:
            self._doc_refs[doc_id] = name
        return row[1]

    def render_text(self, text: str) -> Segment:
        val = self.engine.render_value(text)
        self._render_time = int(time.time())
        log.debug('render_text: %r', val)
        result = to_segment(val)
        log.info(
            'rendered %d -> %s (%s)', len(text), type(result).__name__, self._doc_refs
        )
        return result

    # Exposed as a callback to Bridge, used for `edit_message`.
    async def _update_text(self, seg: Segment) -> int | None:
        if self._update_callback is None:
            log.info('_update_text: no update_callback')
            return

        self._render_time = int(time.time())
        spec = self._format_response(seg)
        r = await self._invoke_update_callback(spec)
        log.debug('_update_text: %s', r)
        if isinstance(r, Message):
            return r.message_id
        if (
            r is not None
            and not isinstance(r, int)
            and (msg := r.get_message()) is not None
        ):
            return msg.message_id

    def render(self, text: str) -> Awaitable[MessageSpec]:
        return self.to_response(self.render_text(text))

    async def to_response(self, rendered: Segment) -> MessageSpec:
        spec = self._format_response(rendered)
        if self._update_callback is not None:
            await self._invoke_update_callback(spec)
        return spec

    def _format_response(self, seg: Segment) -> MessageSpec:
        log.debug('_format_response: %r', seg)
        ctx = self.data
        do_cleanup = get_env_flag(ctx, 'cleanup', True)
        self._as_caption = as_caption = self._as_caption or has_media(ctx)

        if not seg:
            if not as_caption:
                seg = '[empty]'
            text_only = True
        elif isinstance(seg, str):
            if do_cleanup:
                seg = cleanup_text(seg)
            text_only = True
        else:
            text_only = False

        if get_env_flag(ctx, 'dom'):
            from pprint import pformat

            seg = Pre(pformat(seg))
        elif get_env_flag(ctx, 'fold'):
            seg = BlockQuote(seg, expandable=True)
        elif get_env_flag(ctx, 'pre', text_only):
            # Skip wrapping in `Pre` if it's explicitly unset or the content is
            # styled (contains `Element`).
            seg = Pre(seg)

        if self._path is None:
            markup = state = None
        else:
            markup, state = make_markup(
                self._path, ctx, self._markup_state, self._doc_refs
            )

        if get_env_flag(ctx, 'plain'):
            parse_mode = None
        elif get_env_flag(ctx, 'html'):
            parse_mode = 'HTML'
        else:
            parse_mode = 'MarkdownV2'

        func = get_renderer(parse_mode)
        out = []

        if self._trusted and (val := get_env(ctx, 'limit')):
            limit = val if isinstance(val, int) else int(to_str(val))
        else:
            limit = (
                MessageLimit.CAPTION_LENGTH
                if as_caption
                else MessageLimit.MAX_TEXT_LENGTH
            )

        fmt = Formatter(limit)
        for part in self._format_seg(fmt, seg, state):
            log.debug('_format_response: part: %r', part)
            func(part, out)

        seg_num = len(fmt.segments)
        seg_len = fmt.length
        seg_typ = type(seg).__name__
        if isinstance(seg, (list, tuple, str)):
            seg_typ += f'[{len(seg)}]'
        elif isinstance(seg, Element):
            seg_typ += f'[{type(seg.inner).__name__}]'
        del fmt, seg

        result = ''.join(out)
        out_len = len(out)
        res_len = len(result)
        del out

        if get_env_flag(ctx, 'raw'):
            # Markup characters are counted in length when `parse_mode` is unset.
            result = truncate_text(result)
            parse_mode = None

        if do_cleanup:
            result = cleanup_text(result)

        res_len_2 = len(result)
        log.debug('_format_response: final: %r (%d)', result, res_len_2)
        log.info(
            'formatted %s into %d/%d as %d/%d%s, %s, %s',
            seg_typ,
            seg_num,
            seg_len,
            out_len,
            res_len,
            '' if res_len_2 == res_len else f'/{res_len_2}',
            parse_mode or '/',
            markup and len(markup.inline_keyboard) or '/',
        )
        return result, parse_mode, markup

    def _format_seg(
        self, fmt: Formatter, body: Segment, state: str | None
    ) -> Iterable[Segment]:
        # Reserve space for the overflow indicator.
        length = fmt.to_length(body)
        overflow_line = [str(length), OVERFLOWED_TEXT]
        fmt.try_append(Element(overflow_line))

        # Render the footers first, so we can truncate the body if overflowed.
        if errors := self.engine.errors:
            fmt.try_append('\n\n---\n\n')
            for e in errors:
                fmt.try_push(e, '\n')

        is_first = True
        if not fmt.full:
            for seg in self._format_footers(state):
                if is_first:
                    fmt.try_append(seg)
                    is_first = False
                else:
                    fmt.try_push('\n', seg)

        sep = len(fmt.segments)
        footer_len = fmt.length

        if fmt.try_append(body):
            omitted = 0
        else:
            omitted = length + footer_len - fmt.length

        if footer_len == fmt.length:
            if not self._as_caption:
                fmt.try_append('[empty?]' if is_first else '[empty?]\n')
        elif not is_first and not isinstance(fmt.segments[-1], (Pre, BlockQuote)):
            # Add a newline between the body and the footers, unless the body ends
            # with a block (which already implies a line break).
            fmt.try_append('\n')

        if fmt.full:
            if omitted:
                overflow_line[0] = str(omitted)
            else:
                overflow_line.pop(0)
        else:
            overflow_line.clear()

        log.debug(
            '_format_seg: sep=%d body=%d footer=%d fmt=%d omitted=%d full=%s',
            sep,
            length,
            footer_len,
            fmt.length,
            omitted,
            fmt.full,
        )

        # Rotate the body before the footers!
        return chain(islice(fmt.segments, sep, None), islice(fmt.segments, sep))

    def _format_footers(self, state: str | None) -> Iterable[Segment]:
        ctx = self.data

        if footer := get_env(ctx, 'footer'):
            yield to_segment(footer)

        if get_env_flag(ctx, 'show_state', True) and state:
            yield state

        if get_env_flag(ctx, 'show_source') and (source := self.engine.get_doc()):
            yield Pre(source, lang='c')

        if get_env_flag(ctx, 'show_stats', True) and (unix := self._render_time):
            yield (
                Time(str(unix), unix, format='wdt'),
                ' (',
                Time('now', unix, format='r'),
                f') | {self.engine.gas_used()}',
            )


@command(public=True)
def handle_render(msg: Message, rs: Responder, arg: MessageArg):
    target = msg.reply_to_message
    text = target and (target.text or target.caption or '').strip()

    if text:
        if arg:
            text = arg + '\n' + text
    elif arg:
        text = arg
    else:
        return rs.reply_cached('Specify text or reply to a message to render.')

    if doc_ref := is_doc_ref(text):
        path, row = doc_ref
        if path is None:
            return rs.reply_cached(f'No doc: {row}')
        assert isinstance(row, tuple)
        doc_id, text = row
    else:
        chat_prefix = encode_id(msg.chat_id, '')
        path = f'#{chat_prefix}{msg.message_id}'
        db['r-' + path] = text
        doc_id = None

    ctx = RenderContext(doc_id=doc_id, path=path, responder=rs)
    ctx.set_update_callback(create_reply_callback(rs, ctx.data))
    return ctx.render(text)


type AllowedMedia = PhotoPayload | DocumentPayload


def _extract_media(val: Value, typ: Type[AllowedMedia]) -> AllowedMedia | None:
    if isinstance(val, LocalPath):
        content = typ(Path(val.path))
    elif isinstance(val, bytes):
        content = typ(val)
    else:
        return None
    log.info('render: Media %s: %s', typ.__name__, repr(val)[:50])
    return content


def extract_media(data: Mapping[str, Value]):
    if photo := get_env(data, 'photo'):
        r = _extract_media(photo, PhotoPayload)
        if r is not None:
            return r

    if document := get_env(data, 'document'):
        r = _extract_media(document, DocumentPayload)
        if r is not None:
            return r


def has_media(data: Mapping[str, Value]) -> bool:
    return bool(get_env(data, 'photo') or get_env(data, 'document'))


def create_reply_callback(rs: Responder, data: Mapping[str, Value]) -> UpdateCallback:
    def do_reply(spec: MessageSpec):
        return rs.reply_cached(
            *spec, media=extract_media(data), allow_not_modified=True
        )

    return do_reply


DOC_SEARCH_PATH = list_env('DOC_SEARCH_PATH', ':')
DOC_OVERRIDE_DIR = os.environ.get('DOC_OVERRIDE_DIR')
ALLOWED_GUEST_DOC_PREFIXES = list_env('ALLOWED_GUEST_DOC_PREFIXES')

REG_DOC_REF = re.compile(r'[*:]\s*(\w+)\s*;')


def get_doc(name: str, trusted: bool | None = None) -> tuple[int | None, str] | None:
    context = get_context()
    if context.sender_is_guest():
        if any(name.startswith(prefix) for prefix in ALLOWED_GUEST_DOC_PREFIXES):
            log.info('get_doc: allowed guest access to doc: %s', name)
        else:
            log.info('get_doc: disallowed guest access to doc: %s', name)
            return None

    if trusted is None:
        trusted = context.sender_is_host()
    trusted = trusted and os.path.basename(name) == name

    if not trusted:
        return db.get_doc(name)

    if DOC_OVERRIDE_DIR and os.path.isfile(
        file := os.path.join(DOC_OVERRIDE_DIR, name + '.m')
    ):
        with open(file, encoding='utf-8') as fp:
            text = fp.read()
        log.info('get_doc: loaded doc %s from override dir', name)
        return None, text

    row = db.get_doc(name)

    if row is None and DOC_SEARCH_PATH:
        log.info(
            'get_doc: searching doc %s, trusted=%s in %r',
            name,
            trusted,
            DOC_SEARCH_PATH,
        )
        for d in DOC_SEARCH_PATH:
            for ext in ('.m', '.txt'):
                if os.path.isfile(file := os.path.join(d, name + ext)):
                    with open(file, encoding='utf-8') as fp:
                        text = fp.read()
                    return None, text
    return row


def is_doc_ref(
    text: str,
) -> tuple[str, tuple[int | None, str]] | tuple[None, str] | None:
    '''Returns (path, (doc_id | None, text)), (None, doc_name), or None.'''
    if m := REG_DOC_REF.fullmatch(text):
        doc_name = m[1]
        row = get_doc(doc_name)
        log.info('handle_render: doc ref: %s %s', text, row)
        if row is None:
            return None, doc_name
        return ':' + doc_name, row


preview_cache: dict[int, tuple[RenderContext, str, Segment]] = {}


async def handle_render_group(rs: Responder, origin_id: int):
    msg_id = rs.get_message().message_id
    if cache := preview_cache.pop(origin_id, None):
        ctx, doc_name, result = cache
        log.info('Doc in group: %s -> %s %s', msg_id, origin_id, doc_name)

        ctx.set_path(':' + doc_name)
        ctx.set_update_callback(create_reply_callback(rs, ctx.data))
        await ctx.to_response(result)
    else:
        log.info('No preview cache in group: %s -> %s', msg_id, origin_id)


async def handle_render_inline_query(query: InlineQuery, text: str):
    if doc_ref := is_doc_ref(text):
        path, row = doc_ref
        if path is None:
            assert isinstance(row, str)
            msg = 'No doc: ' + row
            r = InlineQueryResultArticle(
                id='0',
                title=msg,
                input_message_content=InputTextMessageContent(msg),
            )
            await query.answer((r,))
            return
        assert isinstance(row, tuple)
        doc_id, text = row
    elif len(text) + 2 < InlineKeyboardButton.MAX_CALLBACK_DATA:
        path = '`' + text
        doc_id = None
    else:
        doc_id = path = None

    ctx = RenderContext(doc_id=doc_id, path=path)
    ctx.data['_env.footer'] = Element((Bold('via '), '@', bot.username, ' ', text))
    rendered = ctx.render_text(text)
    result, parse_mode, markup = await ctx.to_response(rendered)

    if rendered := render_segment(rendered).strip():
        p = 50
        left_chars = len(rendered) - p * 2
        title = (
            (f'{rendered[:p]} ...[{left_chars} chars]... \\{rendered[-p:]}')
            if left_chars > 0
            else rendered
        )
    else:
        title = '[empty]'

    r = InlineQueryResultArticle(
        id='noop',
        title=title,
        input_message_content=InputTextMessageContent(result, parse_mode=parse_mode),
        reply_markup=markup,
    )
    await query.answer((r,))


@callback_query(filter=lambda data: data[0] in CALLBACK_SPECIAL, public=True)
def handle_render_callback(callback: CallbackQuery, data: CallbackData, rs: Responder):
    flags = {}
    clicked_button = None

    j = 0
    sign = ''

    def take_until(i: int, delims: Container[str]) -> str:
        nonlocal j
        j = i + 1
        while j < len(data) and data[j] not in delims:
            j += 1
        return data[i + 1 : j]

    true = CALLBACK_SIGNS[True]
    while (i := j) < len(data) and (sign := data[i]) in CALLBACK_SIGNS:
        key = take_until(i, CALLBACK_SPECIAL)
        if sign == BUTTON_SIGN:
            log.debug('handle_render_callback: button %r', clicked_button)
            if clicked_button is not None:
                raise ValueError('handle_render_callback: multiple buttons: ' + data)
            clicked_button = key
        else:
            flags[key] = sign == true

    if sign == MEMORY_SIGN:
        memory = take_until(i, PATH_SIGNS)
        i = j
        log.debug('handle_render_callback: memory %r', memory)
    else:
        memory = None

    if len(path := data[i:]) <= 2:
        raise ValueError('bad path in render callback: ' + data)

    # Compatibility with old format
    if path[0] == ':' and path[1] in CALLBACK_SPECIAL:
        path = path[1:]

    doc_id = None
    match path[0]:
        case '`':
            text = path[1:]
        case '#':
            text = db.get('r-' + path)
            if text is None:
                raise ValueError('unknown msg in render callback: ' + path)
        case ':':
            row = get_doc(path[1:])
            if row is None:
                raise ValueError('unknown doc in render callback: ' + path)
            doc_id, text = row
        case _:
            raise ValueError('bad render callback: ' + data)

    as_caption = bool(rs.get_message().caption)
    ctx = RenderContext(
        overrides=dict(flags),
        markup_state=(flags, data),
        doc_id=doc_id,
        path=path,
        as_caption=as_caption,
        responder=rs,
    )
    inner = ctx.data
    ctx.set_update_callback(create_callback_query_callback(callback, as_caption, inner))
    if clicked_button is not None:
        inner[BUTTON_KEY] = clicked_button
    if memory is not None:
        inner[MEMORY_KEY] = decode_value(memory)
    inner['_state'] = data
    return ctx.render(text)


def create_callback_query_callback(
    callback: CallbackQuery, as_caption: bool, data: Mapping[str, Value]
) -> UpdateCallback:
    answered = False

    async def edit_callback_message(spec: MessageSpec):
        nonlocal answered, as_caption

        text, parse_mode, markup = spec
        if (media := extract_media(data)) is not None:
            with media.as_input(text, parse_mode) as input_media:
                r = await callback.edit_message_media(input_media, reply_markup=markup)
            as_caption = True
        elif as_caption:
            r = await try_send_text_or_not_modified(
                callback.edit_message_caption,
                text,
                parse_mode=parse_mode,
                reply_markup=markup,
            )
        else:
            r = await try_send_text_or_not_modified(
                callback.edit_message_text,
                text,
                parse_mode=parse_mode,
                reply_markup=markup,
            )

        if (not answered) and (answer := get_env(data, 'answer')):
            answered = True
            answer = to_str(answer)
            if len(answer) > CallbackQuery.MAX_ANSWER_TEXT_LENGTH:
                answer = answer[: CallbackQuery.MAX_ANSWER_TEXT_LENGTH - 1] + '…'

            show_alert = get_env_flag(data, 'answer_alert')
            log.info('handle_render_callback: answer: %s, %s', answer, show_alert)
            await callback.answer(answer, show_alert=show_alert)

        return r

    return edit_callback_message


def _report(
    out: list[str], action: str, id: int | None, name: str | None, text: str | None
):
    parts = []
    if id is not None:
        parts.append(f'{id}:')

    if name is not None:
        parts.append(name)

    if text is not None:
        short = shorten(text)
        parts.append(f'({short})')
        log_info = ' '.join(parts)
        msg_info = escape(' '.join(parts[:-1])) + rf' \(`{escape(short)}`\)'
    else:
        log_info = msg_info = ' '.join(parts)

    log.info('%s %s', action, log_info)

    if id is not None:
        msg_info = f'[{msg_info}]({get_msg_url(id)})'

    out.append(f'{escape(action)} {msg_info}')


def cleanup_preview_cache(doc_id: int, val_id: int, name: str):
    if id(preview_cache.get(doc_id)) == val_id:
        log.warning('Unused preview cache for doc: %s %s', doc_id, name)
        del preview_cache[doc_id]


async def handle_render_doc(msg: Message):
    if not (text := msg.text) or not (text := text.strip()):
        return

    id = msg.message_id
    ctx = RenderContext(doc_id=id)
    result = ctx.render_text(text)

    info = []
    if name := ctx.engine.doc_name:
        preview_cache[id] = t = (ctx, name, result)
        asyncio.get_event_loop().call_later(
            30, cleanup_preview_cache, id, builtins.id(t), name
        )

        old_by_id, old_by_name = db.save_doc(id, name, text)

        action = 'new doc:'
        if old_by_id:
            old_name, old_text = old_by_id
            if old_name == name:
                if old_text == text:
                    old_text = None
                _report(info, 'updated doc:', None, None, old_text)
            else:
                _report(info, 'renamed doc:', None, old_name, old_text)
            action = '->'

        if old_by_name:
            old_id, old_text = old_by_name
            if old_text == text:
                old_text = None
            _report(info, 'relinked doc:', old_id, None, old_text)
            assert old_id != id
            action = '->'

        _report(info, action, id, name, text)
        set_reaction = msg.set_reaction(ReactionEmoji.RED_HEART, True)
    elif old_row := db.delete_doc(id):
        _report(info, 'deleted doc:', id, *old_row)
        set_reaction = msg.set_reaction()
    else:
        return

    res = truncate_text(('\n' if len(info) > 2 else ' ').join(info))
    await asyncio.gather(do_notify(res, 'MarkdownV2', quiet=True), set_reaction)
    return name


@command
def handle_ls(rs: Responder, arg: MessageArg):
    keywords = arg.split()

    if not (docs := tuple(db.find_docs(keywords))):
        return rs.reply_cached('No docs found.')

    lines = [rf'{len(docs)} docs:']
    for id, name, length in docs:
        line = rf'\- [*{escape(name)}*]({get_msg_url(id)}) \({id}, {length}\)'
        lines.append(line)

    return rs.reply_cached('\n'.join(lines), parse_mode='MarkdownV2')


@command
async def handle_rm(rs: Responder, arg: MessageArg):
    names = arg.split()
    if not names:
        return await rs.reply_cached('Usage: /rm <name1> [name2 ...]')
    db.delete_docs(names)
    return await rs.reply_cached(f'Deleted {len(names)} docs.')


@command
async def handle_submit(rs: Responder, arg: MessageArg):
    if not DOC_OVERRIDE_DIR:
        return await rs.reply_cached('DOC_OVERRIDE_DIR is unset.')

    if os.path.basename(arg) != arg:
        return await rs.reply_cached('Bad name.')

    file = os.path.join(DOC_OVERRIDE_DIR, arg + '.m')
    with open(file, encoding='utf-8') as fp:
        text = fp.read()

    length = len(text)
    if length > MAX_TEXT_LENGTH:
        return await rs.reply_cached(f'Too long: {length}')

    m = await bot.send_message(CHAN_ID, *pre_block(text))
    name = await handle_render_doc(m) or arg

    if (r := db.get_doc(name)) is not None:
        old_url = get_msg_url(r[0])
    else:
        old_url = None

    dst = file + '.old'
    if os.path.exists(dst):
        n = 1
        while os.path.exists(dst := f'{file}.old.{n}'):
            n += 1

    os.rename(file, dst)
    log.info('Renamed %s -> %s', file, dst)

    info = f'{length} chars ({name})\nNew: {get_msg_url(m.message_id)}'
    if old_url is not None:
        info = f'{info}\nOld: {old_url}'

    await rs.reply_cached(info)
