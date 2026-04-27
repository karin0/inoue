import os
import re
import time
import asyncio
from itertools import chain, islice
from typing import Container, Iterable, Mapping, Callable, Awaitable, cast

from telegram import (
    CallbackQuery,
    InlineQuery,
    InlineQueryResultArticle,
    InputTextMessageContent,
    Update,
    Message,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import ContextTypes
from telegram.error import BadRequest
from telegram.constants import ReactionEmoji

from db import db
from util import (
    USER_ID,
    DOC_SEARCH_PATH,
    log,
    get_msg_arg,
    get_msg_url,
    reply_text,
    try_send_text,
    shorten,
    truncate_text,
    escape,
    cleanup_text,
    do_notify,
    encode_chat_id,
)
from segments import (
    Segment,
    Element,
    Pre,
    Time,
    BlockQuote,
    Formatter,
    get_renderer,
    render_segment,
)
from render_core import Engine, Value
from render_bridge import Bridge, to_segment
from render_context import OverriddenDict, encode_value, decode_value

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


def get_env_flag[T](
    ctx: Mapping[str, Value], key: str, default: T = False
) -> Value | T:
    v = get_env(ctx, key)
    if v is None:
        return default
    if v == '0':
        return False
    if v == '1':
        return True
    return v


def make_markup(
    path: str,
    ctx: Mapping[str, Value],
    current_state: MarkupState | None,
    doc_id: int | None,
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

    if doc_id is not None and not get_env_flag(ctx, 'ref', True):
        doc_id = None

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

            if icon := get_env(ctx, 'icon.' + k):
                label = str(icon)
                push_flag(label + (':=' if old else '=') + '01'[v])
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
        row.append(InlineKeyboardButton('🔄', callback_data=current_data))

    if doc_id:
        row.append(InlineKeyboardButton('🔗', get_msg_url(doc_id)))

    if len(row) <= 1:
        return None, None

    row[0] = InlineKeyboardButton(
        ('⏪ ' if current_data else '🔄 ') + shorten(path[1:]), callback_data=path
    )
    row_ = cast(list[InlineKeyboardButton], row)

    row_limit_ = get_env(ctx, 'btns_per_row', 5)
    try:
        if isinstance(row_limit_, (str, int, float)):
            row_limit = int(row_limit_)
        else:
            row_limit = 5
    except (ValueError, TypeError):
        row_limit = 5

    # Group 5 buttons per row
    rows = [row_[i : i + row_limit] for i in range(0, len(row), row_limit)]
    return InlineKeyboardMarkup(rows), current_data


type MessageSpec = tuple[str, str | None, InlineKeyboardMarkup | None]
type UpdateCallback = Callable[[MessageSpec], Awaitable[Message | bool | None]]
type MarkupState = tuple[dict[str, bool], str]

OVERFLOWED_TEXT = '…\n'


class RenderContext:
    __slots__ = (
        '_markup_state',
        '_doc_id',
        '_path',
        '_update_callback',
        '_first_doc_id',
        '_render_time',
        '_trusted',
        'data',
        'engine',
    )

    def __init__(
        self,
        update: Update,
        overrides: dict[str, Value] | None = None,
        markup_state: MarkupState | None = None,
        doc_id: int | None = None,
        path: str | None = None,
        update_callback: UpdateCallback | None = None,
    ):
        self._markup_state = markup_state
        self._doc_id = doc_id
        self.set_path(path)
        self.set_update_callback(update_callback)

        if overrides is None:
            overrides = {}

        # We assume content from saved docs and USER_ID is trusted.
        trusted = doc_id
        source = None

        # Existing `overrides` are frozen and immutable in `Engine`, so this is safe.
        if user := update.effective_user:
            overrides['_user_id'] = user.id
            overrides['_user_name'] = source = user.full_name
        if chat := update.effective_chat:
            if trusted is None and chat.id == USER_ID:
                trusted = USER_ID
            overrides['_chat_id'] = chat.id
            if title := chat.title:
                overrides['_chat_title'] = title
                source = f'{title} @ {source}' if source else title
        if source:
            overrides['_source'] = source
        if msg := update.effective_message:
            overrides['_msg_id'] = msg.message_id
        if trusted is not None:
            overrides['_trusted'] = trusted

        log.info('create_engine: %s', overrides)

        self._first_doc_id = None
        self._render_time = None
        self._trusted = trusted

        self.data = OverriddenDict({}, overrides)
        bridge = Bridge(self.data, self._update_text, trusted)

        # Access to attributes with underscores should be forbidden in `simpleeval`,
        # so `os` is safe.
        self.data['os'] = self.data['sys'] = bridge

        self.engine = Engine(self.data, self._doc_loader, funcs=bridge.funcs)

    def set_path(self, path: str | None) -> None:
        self._path = path

    def set_update_callback(self, callback: UpdateCallback | None) -> None:
        self._update_callback = callback and (callback, asyncio.Lock())

    async def _invoke_update_callback(self, spec: MessageSpec) -> Message | bool | None:
        if self._update_callback is not None:
            func, lock = self._update_callback
            async with lock:
                return await func(spec)

    def _doc_loader(self, name: str) -> str | None:
        row = db.get_doc(name)
        if row is None:
            if self._trusted and DOC_SEARCH_PATH and os.path.basename(name) == name:
                for d in DOC_SEARCH_PATH:
                    if os.path.isfile(
                        file := os.path.join(d, name + '.m')
                    ) or os.path.isfile(file := os.path.join(d, name + '.txt')):
                        with open(file, encoding='utf-8') as fp:
                            return fp.read()
            return None
        if self._first_doc_id is None:
            self._first_doc_id = row[0]
        return row[1]

    def render_text(self, text: str) -> Segment:
        val = self.engine.render_value(text)
        log.debug('render_text: %r', val)
        result = to_segment(val)
        self._render_time = int(time.time())
        log.info(
            'rendered %d -> %s (%s)',
            len(text),
            type(result).__name__,
            self._first_doc_id,
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

    async def render(
        self,
        text: str,
    ) -> MessageSpec:
        rendered = self.render_text(text)
        return await self.to_response(rendered)

    async def to_response(self, rendered: Segment) -> MessageSpec:
        spec = self._format_response(rendered)
        if self._update_callback is not None:
            await self._invoke_update_callback(spec)
        return spec

    def _format_response(self, seg: Segment) -> MessageSpec:
        log.debug('_format_response: %r', seg)
        ctx = self.data
        do_cleanup = get_env_flag(ctx, 'cleanup', True)

        if not seg:
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
            if (doc_id := self._first_doc_id) is None:
                doc_id = self._doc_id
            markup, state = make_markup(self._path, ctx, self._markup_state, doc_id)

        if get_env_flag(ctx, 'plain'):
            parse_mode = None
        elif get_env_flag(ctx, 'html'):
            parse_mode = 'HTML'
        else:
            parse_mode = 'MarkdownV2'

        func = get_renderer(parse_mode)
        out = []

        for seg in self._format_seg(seg, state):
            log.debug('_format_response: seg: %r', seg)
            func(seg, out)

        result = ''.join(out)

        if get_env_flag(ctx, 'raw'):
            # Markup characters are counted in length when `parse_mode` is unset.
            result = truncate_text(result)
            parse_mode = None

        if do_cleanup:
            result = cleanup_text(result)

        log.debug('_format_response: final: %r (%d)', result, len(result))
        return result, parse_mode, markup

    def _format_seg(self, body: Segment, state: str | None) -> Iterable[Segment]:
        fmt = Formatter()

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
            fmt.try_append('[empty?]')
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

        if get_env_flag(ctx, 'show_source') and (source := self.engine.get_doc_text()):
            yield Pre(source, lang='c')

        if get_env_flag(ctx, 'show_stats', True) and (unix := self._render_time):
            yield (
                Time(str(unix), unix, format='wdt'),
                ' (',
                Time('now', unix, format='r'),
                f') | {self.engine.gas_used()}',
            )


REG_DOC_REF = re.compile(r'[*:]\s*(\w+)\s*;')


async def handle_render(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    msg, arg = get_msg_arg(update)
    target = msg.reply_to_message
    text = target and (target.text or target.caption or '').strip()

    if text:
        if arg:
            text = arg + '\n' + text
    elif arg:
        text = arg
    else:
        return await reply_text(msg, 'Specify text or reply to a message to render.')

    if doc_ref := is_doc_ref(text):
        path, row = doc_ref
        if path is None:
            return await reply_text(msg, f'No doc: {text}')
        assert isinstance(row, tuple)
        doc_id, text = row
    else:
        chat_prefix = encode_chat_id(msg, '')
        path = f'#{chat_prefix}{msg.message_id}'
        db['r-' + path] = text
        doc_id = None

    async def edit_reply_message(spec: MessageSpec):
        return await reply_text(msg, *spec, allow_not_modified=True)

    ctx = RenderContext(
        update, doc_id=doc_id, path=path, update_callback=edit_reply_message
    )
    await ctx.render(text)


def is_doc_ref(text: str) -> tuple[str, tuple[int, str]] | tuple[None, str] | None:
    if m := REG_DOC_REF.fullmatch(text):
        doc_name = m[1]
        row = db.get_doc(doc_name)
        log.info('handle_render: doc ref: %s %s', text, row)
        if row is None:
            return None, doc_name
        return ':' + doc_name, row


preview_cache: dict[int, tuple[RenderContext, str, Segment]] = {}


async def handle_render_group(msg: Message, origin_id: int):
    if cache := preview_cache.pop(origin_id, None):
        ctx, doc_name, result = cache
        log.info('Doc in group: %s -> %s %s', msg.id, origin_id, doc_name)

        async def edit_reply_message(spec: MessageSpec):
            return await reply_text(msg, *spec, allow_not_modified=True)

        ctx.set_path(':' + doc_name)
        ctx.set_update_callback(edit_reply_message)
        await ctx.to_response(result)
    else:
        log.info('No preview cache in group: %s -> %s', msg.id, origin_id)

    if preview_cache:
        left = tuple((id, name) for id, (_, name, _) in preview_cache.items())
        log.warning('Preview cache not empty: %s', left)
        preview_cache.clear()


async def handle_render_inline_query(update: Update, query: InlineQuery, text: str):
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
            return await query.answer((r,))
        assert isinstance(row, tuple)
        doc_id, text = row
    elif len(text) + 2 < InlineKeyboardButton.MAX_CALLBACK_DATA:
        path = '`' + text
        doc_id = None
    else:
        doc_id = path = None

    ctx = RenderContext(update, doc_id=doc_id, path=path)
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


async def handle_render_callback(update: Update, query: CallbackQuery, data: str):
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
                # TODO: report when cache expired
                raise ValueError('unknown msg in render callback: ' + path)
        case ':':
            row = db.get_doc(path[1:])
            if row is None:
                # TODO: report when doc deleted
                raise ValueError('unknown doc in render callback: ' + path)
            doc_id, text = row
        case _:
            raise ValueError('bad render callback: ' + data)

    async def edit_callback_message(spec: MessageSpec):
        text, parse_mode, markup = spec
        try:
            return await try_send_text(
                query.edit_message_text,
                text,
                parse_mode=parse_mode,
                reply_markup=markup,
            )
        except BadRequest as e:
            if 'Message is not modified' not in str(e):
                raise

    ctx = RenderContext(
        update,
        overrides=dict(flags),
        markup_state=(flags, data),
        doc_id=doc_id,
        path=path,
        update_callback=edit_callback_message,
    )
    if clicked_button is not None:
        ctx.data[BUTTON_KEY] = clicked_button
    if memory is not None:
        ctx.data[MEMORY_KEY] = decode_value(memory)
    ctx.data['_state'] = data

    await ctx.render(text)


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


async def handle_render_doc(update: Update, msg: Message):
    if not (text := msg.text) or not (text := text.strip()):
        return

    id = msg.message_id
    ctx = RenderContext(update, doc_id=id)
    result = ctx.render_text(text)

    info = []
    if name := ctx.engine.doc_name:
        preview_cache[id] = (ctx, name, result)
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
    await asyncio.gather(
        do_notify(res, 'MarkdownV2', quiet=True),
        set_reaction,
    )


async def handle_ls(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg, arg = get_msg_arg(update)
    keywords = arg.split()

    if not (docs := tuple(db.find_docs(keywords))):
        return await reply_text(msg, 'No docs found.')

    lines = [rf'{len(docs)} docs:']
    for id, name, length in docs:
        line = rf'\- [*{escape(name)}*]({get_msg_url(id)}) \({id}, {length}\)'
        lines.append(line)

    await reply_text(msg, '\n'.join(lines), parse_mode='MarkdownV2')
