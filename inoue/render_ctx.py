import asyncio
import os
import re
import time
import weakref

from collections.abc import Awaitable, Callable, Iterable, Mapping
from itertools import chain, islice
from pathlib import Path

from telegram import InlineKeyboardMarkup
from telegram.constants import MessageLimit

from bot import (
    DocumentPayload,
    EditHandle,
    PhotoPayload,
    Responder,
    create_task,
    shorten,
    truncate_text,
)
from render_core import Box, Engine, Value, to_str

from .ctx import get_context
from .db import db
from .env import CHAN_ID, USER_ID, list_env
from .log import log
from .render_bridge import Bridge, count_tasks
from .render_context import OverriddenDict
from .render_lib import FlattenSegment, LocalPath, merge_segments, to_segment
from .segments import BlockQuote, Element, Formatter, Pre, Segment, Time, get_renderer
from .text import cleanup_text

ENV_PREFIX = '_env.'
BUTTON_KEY = '_btn'


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


type MessageSpec = tuple[str, str | None, InlineKeyboardMarkup | None]
type UpdateCallback = Callable[[MessageSpec], Awaitable[EditHandle]]
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
        '_edit_handle',
        '_error_idx',
        '_editing',
        'data',
        'engine',
        '__weakref__',
    )

    def __init__(
        self,
        data: OverriddenDict,
        markup_state: MarkupState | None = None,
        as_caption: bool = False,
        doc_id: int | None = None,
        path: str | None = None,
        update_callback: UpdateCallback | None = None,
        responder: Responder | None = None,
    ):
        self.data = data
        self._markup_state = markup_state
        self._as_caption = as_caption
        self._responder = rs = responder
        self.set_path(path)
        self.set_update_callback(update_callback)

        self._doc_refs = {doc_id: ''} if doc_id is not None else {}
        self._render_time = None

        self._error_idx = 0
        self._edit_handle = None
        self._editing: tuple[list[FlattenSegment], asyncio.Future[str | None]] | None = None

        context = get_context()
        if (sender := context.sender) is not None and sender.id in (USER_ID, CHAN_ID):
            data.overrides['_trusted'] = trusted = sender.id
        else:
            trusted = None

        # We assume contents from USER_ID and saved docs are trusted.
        # However, docs (typically in ALLOWED_GUEST_DOC_PREFIXES) need to escalate
        # explicitly if expanded from a guest context, declaring that their security
        # does not rely on any context values, which can be spoofed by guests via
        # callback data.
        self._trusted = trusted
        self._can_escalate = trusted or doc_id

        if not (path and update_callback and rs):
            task_policy = 'uneditable context'
        elif self.data.get(BUTTON_KEY) == '_cancel':
            task_policy = 'context cancelled'
        else:
            task_policy = None

        bridge = Bridge(
            data, trusted, task_policy, rs.get_message_key() if rs is not None else None, self
        )

        # Access to attributes with underscores should be forbidden in `simpleeval`,
        # so `os` is safe.
        self.data['os'] = self.data['sys'] = bridge

        this = weakref.ref(self)

        def doc_loader(name: str) -> str | None:
            return this()._doc_loader(name)  # pyright: ignore[reportOptionalMemberAccess]

        self.engine = Engine(self.data, doc_loader, funcs=bridge._get_func)

    def _error(self, msg: str) -> None:
        self.engine.errors.append(msg)

    def set_path(self, path: str | None) -> None:
        self._path = path

    def set_update_callback(self, callback: UpdateCallback | None) -> None:
        self._update_callback = callback and (callback, asyncio.Lock())

    async def _invoke_update_callback(self, seg: FlattenSegment) -> MessageSpec:
        assert self._update_callback is not None
        editing = self._editing
        self._editing = None
        func, lock = self._update_callback
        async with lock:
            if editing is not None:
                content, fut = editing
                if fut.cancelled():
                    log.debug('_invoke_update_callback: context cancelled: %r', seg)
                    return self._format_response(seg)
                seg = merge_segments((seg, *content)) if seg else merge_segments(content)
            spec = self._format_response(seg)
            self._edit_handle = handle = await func(spec)
            key = handle.get_message_key()
            log.debug('_invoke_update_callback: key=%r, editing=%r', key, editing)
            if editing is not None:
                _, fut = editing
                if fut.cancelled():
                    log.info('_invoke_update_callback: context cancelled after update: %r', fut)
                else:
                    fut.set_result(key)
            return spec

    def _escalate(self) -> int | None:
        if (token := self._can_escalate) is not None:
            log.info('Escalating privileges: %r %r', self._trusted, token)
            self._trusted = token
            self.data.overrides['_trusted'] = token
            return token

        log.warning('Escalation rejected: %r', self._trusted)

    def __repr__(self):
        data = self.data
        doc = self.engine.get_doc()
        if doc is None:
            doc = '/'
        else:
            # Clean up indentation.
            doc = re.sub(r'\s+', ' ', doc) if doc else ''
            doc = repr(shorten(doc, 80))
        rs = self._responder
        text = '/' if rs is None else repr(shorten(rs.get_text(), 80).replace('\n', ' '))
        return (
            f'RenderContext: {data.get("_source", "?")} ({data.get("_chat_id", "?")}, '
            f'{data.get("_msg_id", "?")}): {self._path}\n {doc}\n {text}'
        )

    def _doc_loader(self, name: str) -> str | None:
        row = get_doc(name, bool(self._trusted))
        if row is None:
            return None
        if (doc_id := row[0]) is not None:
            self._doc_refs[doc_id] = name
        return row[1]

    def render_text(self, text: str) -> FlattenSegment:
        val = self.engine.render_value(text)
        self._atexit()
        log.debug('render_text: %r', val)
        seg = to_segment(val)
        log.info('rendered %d -> %s (%s)', len(text), type(seg).__name__, self._doc_refs)
        return seg

    def _atexit(self) -> Awaitable[MessageSpec | None] | MessageSpec | None:
        if callable(hook := get_env(self.data, 'atexit')):
            if isinstance(hook, Box):
                log.debug('_atexit: calling hook: %r', hook)
                # Calling sub-docs (or Engine) should never raise exceptions.
                hook()
            else:
                raise TypeError(f'hook is not a Box: {hook!r}')
        self._render_time = int(time.time())

    # Exposed as a callback to Bridge, used for `edit_message`.
    # This does not count as a last task in `count_tasks`.
    def _edit_message(self, val: Value | None) -> asyncio.Future[str | None]:
        if self._update_callback is None:
            raise RuntimeError('uneditable context')

        # We always append the new value, so the doc can clear the message by editing it to empty.
        seg = to_segment(val) if val is not None else ''

        # To gather multiple edits, along with all `engine.errors` generated during the task
        # callback, we have to defer the actual update until `_task_done` is called.
        if self._editing is None:
            fut = asyncio.get_event_loop().create_future()
            self._editing = ([seg], fut)
        else:
            content, fut = self._editing
            content.append(seg)

        log.debug('_edit_message: %r', self._editing)
        return fut

    def _task_done(self):
        self._atexit()
        log.debug(
            '_task_done: editing=%r errors=%d/%d',
            self._editing,
            self._error_idx,
            len(self.engine.errors),
        )

        if self._editing is not None:
            log.debug('_task_done: _editing: %r', self._editing)
            # The edit buffer is only preserved during a single task callback.
            create_task(self._invoke_update_callback(''))
        elif (errors := self.engine.errors) and self._error_idx < len(errors):
            log.debug('_task_done: new errors: %d/%d', self._error_idx, len(errors))
            # If new error occurs without calling `_edit_message`, we still want to
            # reply it to the user when possible.
            create_task(self._report_errors())

    async def _report_errors(self):
        if (rs := await self._reply_to_rs()) is not None:
            errors = self.engine.errors
            new_errors = errors[self._error_idx :]
            self._error_idx = len(errors)
            log.info('_task_done: flushing %d errors: %r', len(new_errors), new_errors)
            await rs.reply('\n'.join(new_errors))

    async def _reply_to_rs(self) -> Responder | None:
        # We do not just use `self._responder`, which might override our original reply when
        # handling a callback from an inline message.
        # This ensures we only reply to our rendered message with a new one.
        assert self._update_callback is not None
        _, lock = self._update_callback

        # Wait until any ongoing update that provides the handle is done.
        async with lock:
            h = self._edit_handle
            if h is not None and (h := h.as_responder()) is not None:
                return h

    async def _reply(self, val: Value | None) -> str | None:
        assert self._responder
        seg = to_segment(val) if val is not None else ''
        spec = self._format_response(seg, has_markup=False)
        rs = (await self._reply_to_rs()) or self._responder
        h = await rs.reply(*spec)
        log.debug('_reply: replied: %r', h and h.get_message_key())
        return h.get_message_key() if h is not None else None

    def render(self, text: str) -> Awaitable[MessageSpec]:
        return self.to_response(self.render_text(text))

    async def to_response(self, seg: FlattenSegment) -> MessageSpec:
        # Bootstrapping path.
        if self._update_callback is not None:
            return await self._invoke_update_callback(seg)
        return self._format_response(seg)

    def _format_response(self, seg: FlattenSegment, *, has_markup: bool = True) -> MessageSpec:
        from .render import make_markup

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

        if self._path is None or not has_markup:
            markup = state = None
        else:
            rs = self._responder
            ext_btn = (
                (f'🛑{c}', '_cancel')
                if rs is not None and (c := count_tasks(rs.get_message_key()))
                else None
            )
            markup, state = make_markup(
                self._path, ctx, self._markup_state, self._doc_refs, ext_btn
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
            limit = MessageLimit.CAPTION_LENGTH if as_caption else MessageLimit.MAX_TEXT_LENGTH

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

    def _format_seg(self, fmt: Formatter, body: Segment, state: str | None) -> Iterable[Segment]:
        # Reserve space for the overflow indicator.
        length = fmt.to_length(body)
        overflow_line = [str(length), OVERFLOWED_TEXT]
        fmt.try_append(Element(overflow_line))

        # Render the footers first, so we can truncate the body if overflowed.
        if errors := self.engine.errors:
            self._error_idx = len(errors)
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

        omitted = 0 if fmt.try_append(body) else length + footer_len - fmt.length

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

        if get_env_flag(ctx, 'no_footer'):
            return

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


DOC_SEARCH_PATH = list_env('DOC_SEARCH_PATH', ':')
DOC_OVERRIDE_DIR = os.environ.get('DOC_OVERRIDE_DIR')
ALLOWED_GUEST_DOC_PREFIXES = list_env('ALLOWED_GUEST_DOC_PREFIXES')


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

    if DOC_OVERRIDE_DIR and os.path.isfile(file := os.path.join(DOC_OVERRIDE_DIR, name + '.m')):
        with open(file, encoding='utf-8') as fp:
            text = fp.read()
        log.info('get_doc: loaded doc %s from override dir', name)
        return None, text

    row = db.get_doc(name)

    if row is None and DOC_SEARCH_PATH:
        log.info('get_doc: searching doc %s, trusted=%s in %r', name, trusted, DOC_SEARCH_PATH)
        for d in DOC_SEARCH_PATH:
            for ext in ('.m', '.txt'):
                if os.path.isfile(file := os.path.join(d, name + ext)):
                    with open(file, encoding='utf-8') as fp:
                        text = fp.read()
                    return None, text
    return row


type AllowedMedia = PhotoPayload | DocumentPayload


def _extract_media(val: Value, typ: type[AllowedMedia]) -> AllowedMedia | None:
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
