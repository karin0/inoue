import re
import string
import asyncio

from typing import Any, Iterable, Callable
from collections import UserDict

from simpleeval import simple_eval

from telegram import (
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

from util import (
    log,
    get_msg_arg,
    get_msg_url,
    reply_text,
    shorten,
    truncate_text,
    escape,
    pre_block,
    do_notify,
)
from db import db

CALLBACK_SIGNS = '/+'
ALL_CALLBACK_SIGNS = frozenset('/+:#`')

# Allowed types for context values, as allowed by `simpleeval` by default (except None).
type Value = str | int | float | bool


class OverriddenDict(UserDict):
    def __init__(self, overrides: dict[str, Value]):
        super().__init__()
        self.overrides = overrides

    def __getitem__(self, key: str) -> Value:
        if (v := self.overrides.get(key)) is not None:
            return v
        return self.data[key]

    # For `get` method to work.
    def __contains__(self, key: str) -> bool:
        return key in self.overrides or key in self.data

    def setdefault(self, key: str, default: Value) -> Value:
        # For `?=` operator.
        #
        # We always set the default in the underlying dict, even if the value
        # is overridden, so the natural order of keys is preserved as defined by
        # documents and won't change after markup buttons are activated.
        #
        # For the same purpose, `__setitem__` (for `=` operator) is not overridden.
        r0 = self.data.setdefault(key, default)
        if (r := self.overrides.get(key)) is not None:
            return r
        return r0

    def setdefault_override(self, key: str, value: Value) -> Value:
        # For `:=` operator.
        #
        # This only affects the `overrides` dict, which has higher priority than
        # the underlying dict by their *values*, but are placed after those touched
        # by `=` and `?=` in the natural order of *keys*.
        #
        # There is no need to write back the overridden *value* either here or
        # in `__setitem__`, since `finalize()` already reflects the values from
        # `overrides`.
        return self.overrides.setdefault(key, value)

    def clone(self) -> 'OverriddenDict':
        new = OverriddenDict(self.overrides.copy())
        new.data = self.data.copy()
        return new

    def finalize(self) -> Iterable[tuple[str, Value]]:
        # Keys from the underlying dict are yielded first, in their natural order.
        r = self.data
        r.update(self.overrides)
        return r.items()


def make_markup(
    path: str | None, ctx: 'RenderContext', state: dict[str, bool] | None
) -> InlineKeyboardMarkup | None:
    if not path:
        return None

    # query data: ('/'|'+' <flag-key>)* <path-header: ':'|'#'|'`'> <path-body>
    # where '/' means 0, '+' means 1

    size = len(path.encode('utf-8'))
    if state:
        size += sum(len(k.encode('utf-8')) for k in state.keys()) + len(state)

    if size > InlineKeyboardButton.MAX_CALLBACK_DATA:
        return None

    flags = {}
    for k, v in ctx.ctx.finalize():
        if v == '0' or v == 0:  # This covers `False` as well.
            v = False
        elif v == '1' or v == 1:
            v = True
        else:
            continue

        if size + len(k.encode('utf-8')) + 1 <= InlineKeyboardButton.MAX_CALLBACK_DATA:
            flags[k] = v

    log.debug('make_markup: flags %s state %s', flags, state)

    # The external state takes precedence even over `ctx.overrides`.
    if state:
        flags.update(state)

    if (doc_id := ctx.first_doc_id) is None and not flags:
        return None

    row = [
        InlineKeyboardButton(
            ('⏪ ' if state else '🔄 ') + shorten(path[1:]), callback_data=path
        )
    ]

    if flags:
        if state is None:
            state = {}

        def push_button(name: str):
            data = ''.join((CALLBACK_SIGNS[v] + k) for k, v in state.items()) + path
            row.append(InlineKeyboardButton(name, callback_data=data))

        for k, v in flags.items():
            old = k in state
            state[k] = not v

            push_button(k + (':=' if old else '=') + '01'[v])

            if old:
                state[k] = v
            else:
                del state[k]

        if state:
            push_button('🔄')

    if doc_id:
        row.append(InlineKeyboardButton('📄', get_msg_url(doc_id)))

    # Group 5 buttons per row
    rows = [row[i : i + 5] for i in range(0, len(row), 5)]

    return InlineKeyboardMarkup(rows)


def rendered_response(
    flags: dict[str, bool] | None,
    path: str | None,
    result: str,
    ctx: 'RenderContext',
) -> tuple[str, str | None, InlineKeyboardMarkup | None]:
    result = result or '[empty]'
    if errors := ctx.errors:
        result += f'\n\n---\n\n' + '\n'.join(errors)

    result = truncate_text(result)
    parse_mode = None
    if ctx.get_flag('_pre', True):
        result, parse_mode = pre_block(result)

    return result, parse_mode, make_markup(path, ctx, flags)


def render_text(
    text: str,
    flags: dict[str, bool] | None = None,
    path: str | None = None,
) -> tuple[str, str | None, InlineKeyboardMarkup | None]:
    # Real docs don't need `this_doc`, and we don't know their IDs here.
    this_doc = None if path and path[0] == ':' else (None, text)
    ctx = RenderContext(dict(flags) if flags is not None else {}, this_doc=this_doc)
    result = ctx.render(text)
    log.info('rendered %d -> %d', len(text), len(result))

    if path is None and len(text) + 2 < InlineKeyboardButton.MAX_CALLBACK_DATA:
        path = '`' + text

    return rendered_response(flags, path, result, ctx)


async def handle_render(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    msg, arg = get_msg_arg(update)
    target = msg.reply_to_message

    if target:
        text = (target.text or target.caption).strip()
        if not text:
            return await reply_text(msg, 'No text to render.')

        if arg:
            # Include context set by arg
            text = arg + '\n' + text
    elif arg:
        text = arg
    else:
        return await reply_text(msg, 'Specify text or reply to a message to render.')

    path = f'#{msg.message_id}'
    db['r-' + path] = text
    return await reply_text(msg, *render_text(text, path=path), allow_not_modified=True)


preview_cache = {}


async def handle_render_group(msg: Message, origin_id: int):
    if cache := preview_cache.pop(origin_id, None):
        doc_name, *args = cache
        log.info('Doc in group: %s -> %s %s', msg.id, origin_id, doc_name)
        args = rendered_response(None, ':' + doc_name, *args)
        await reply_text(msg, *args, allow_not_modified=True)

    if left := tuple((id, name) for id, (name, *_) in preview_cache.items()):
        log.warning('Preview cache not empty: %s', left)


async def handle_render_inline_query(query: InlineQuery, text: str):
    result, parse_mode, markup = render_text(text)
    r = InlineQueryResultArticle(
        id='1',
        title=f'Render: {len(result)}',
        input_message_content=InputTextMessageContent(result, parse_mode=parse_mode),
        reply_markup=markup,
    )
    await query.answer((r,))


async def handle_render_callback(
    update: Update, ctx: ContextTypes.DEFAULT_TYPE, data: str
):
    flags = {}
    j = 0
    true = CALLBACK_SIGNS[True]
    while (i := j) < len(data) and (sign := data[i]) in CALLBACK_SIGNS:
        j = i = i + 1
        while j < len(data) and data[j] not in ALL_CALLBACK_SIGNS:
            j += 1
        flags[data[i:j]] = sign == true

    if len(path := data[i:]) <= 2:
        raise ValueError('bad path in render callback: ' + data)

    # Compatibility with old format
    if path[0] == ':' and path[1] in ALL_CALLBACK_SIGNS:
        path = path[1:]

    match path[0]:
        case '`':
            data = path[1:]
        case '#':
            data = db['r-' + path]
        case ':':
            _, data = db.get_doc(path[1:])
        case _:
            raise ValueError('bad render callback: ' + data)

    result, parse_mode, markup = render_text(data, flags, path)

    try:
        if update.callback_query.inline_message_id:
            await ctx.bot.edit_message_text(
                text=result,
                inline_message_id=update.callback_query.inline_message_id,
                reply_markup=markup,
                parse_mode=parse_mode,
            )
        else:
            await ctx.bot.edit_message_text(
                text=result,
                chat_id=update.effective_chat.id,
                message_id=update.callback_query.message.message_id,
                reply_markup=markup,
                parse_mode=parse_mode,
            )
    except BadRequest as e:
        if 'Message is not modified' not in str(e):
            raise


reg = re.compile(r'{(.+?)}|^([^\r\n]+?);$', re.DOTALL | re.MULTILINE)
all_punct = frozenset(string.punctuation.replace('-', '').replace('_', ''))
MAX_DEPTH = 20


def to_str(v: Value) -> str:
    if isinstance(v, bool):
        return '1' if v else '0'
    return str(v)


class RenderContext:
    def __init__(
        self,
        overrides: dict[str, Value],
        *,
        this_doc: tuple[int | None, str] | None = None,
    ):
        self.ctx = OverriddenDict(overrides)
        self.this_doc = this_doc
        self.depth: int = 0
        self.errors: list[str] = []
        self.first_doc_id: int | None = None
        self.doc_name: str | None = None

    def get_ctx(self, key: str, expr: str, *, as_str: bool = False) -> Value:
        if (val := self.ctx.get(key)) is None:
            self.errors.append('undefined: ' + expr)
            return ''

        return to_str(val) if as_str else val

    # Evaluate expression to a value.
    # This should not have side effects on `ctx`.
    def evaluate(
        self, expr: str, *, in_directive: bool = True, as_str: bool = False
    ) -> Value:
        # Empty
        if not (expr := expr.strip()):
            return ''

        # Literal: {`raw}
        if expr[0] == '`':
            return expr[1:]

        if len(expr) >= 2:
            # Script: {"1 + 1"}
            if expr[0] == expr[-1] == '\"':
                expr = expr[1:-1].strip()
                try:
                    v = simple_eval(expr, names=self.ctx.clone())
                except Exception as e:
                    self.errors.append(f'evaluation: {expr}: {type(e).__name__}: {e}')
                    v = ''
                else:
                    log.debug('evaluated script: %s -> %s', expr, v)
                    # Ensure a `Value`.
                    if v is None:
                        return ''

                # Not necessarily str!
                return to_str(v) if as_str else v

            # Literal: {'raw'}
            if expr[0] == expr[-1] == '\'':
                return expr[1:-1]

        # Db doc get: {:name} or {*name}
        if expr[0] == ':' or expr[0] == '*':
            # Recursion is allowed up to a limit.
            if self.depth >= MAX_DEPTH:
                self.errors.append('max depth exceeded: ' + expr)
                return ''

            key = expr[1:].strip()

            # Allow self-reference in `handle_render` and `handle_render_doc`,
            # where the doc is not saved yet.
            # Note that `doc_id` can be None in `handle_render`, where the
            # `doc_name` is transient and only used for recursion.
            if row := self.doc_name == key and self.this_doc or db.get_doc(key):
                doc_id, doc = row
            else:
                self.errors.append('undefined: ' + expr)
                return ''

            if self.first_doc_id is None:
                self.first_doc_id = doc_id

            self.depth += 1
            r = self.render(doc)
            self.depth -= 1
            return r

        # Context equality: {key=value} (by string representation)
        if (p := expr.find('=', 1)) >= 0:
            key = expr[:p].strip()
            val = self.evaluate(expr[p + 1 :].strip(), as_str=True)
            now = self.get_ctx(key, expr, as_str=True)
            return '1' if now == val else '0'

        # Context get: {$key} or {key} (out of a condition directive)
        if expr[0] == '$' or not in_directive:
            key = expr.lstrip('$').strip()
            val = self.get_ctx(key, expr, as_str=as_str)
            return val

        # Literal: {raw}
        return expr

    def _do_assignments(
        self, cmd: str, p1: int, p2: int, set: Callable[[str, Value], Any]
    ):
        key1 = cmd[:p1].strip()
        q = (p := p2) - 1
        while p < len(cmd):
            if (c := cmd[p]) == '=':
                q = p
            elif c in all_punct:
                break
            p += 1
        val = self.evaluate(cmd[q + 1 :].strip())
        set(key1, val)
        if keys := cmd[p2:q].strip():
            for key in keys.split('='):
                set(key.strip(), val)

    def _repl(self, m: re.Match) -> str:
        cmd: str = m[1] or m[2]
        if not (cmd and (cmd := cmd.strip())):
            return m[0]

        # Comment
        if cmd[0] == '#':
            return ''

        ctx = self.ctx
        errors = self.errors

        # Conditional: {cond ? true-directive : false-directive} (false part optional)
        # Omit `?=` which is handled inside directives and cannot occur in conditions
        if in_directive := (
            (p := cmd.find('?', 1)) >= 0 and p + 1 < len(cmd) and cmd[p + 1] != '='
        ):
            # Not `as_str`, so `False` is still treated falsey.
            val = self.evaluate(cmd[:p].strip(), in_directive=False)
            test = val and val != '0'

            # This overrides doc expansion like `? :<doc>`, where we have to
            # use `*<doc>` instead.
            if (q := cmd.find(':', p + 1)) != -1:
                cmd = cmd[p + 1 : q].strip() if test else cmd[q + 1 :].strip()
            elif test:
                cmd = cmd[p + 1 :].strip()
            else:
                return ''

        ret = None

        def handle_cmd(cmd: str):
            nonlocal ret

            # Empty directive
            if not cmd:
                return

            # Literal: {`raw}
            if cmd[0] == '`':
                if ret is None:
                    ret = cmd[1:]
                else:
                    errors.append('redundant literal: ' + cmd)
                return

            # Db name set: {name:}
            if cmd[-1] == ':':
                if self.depth == 0:
                    if self.doc_name:
                        errors.append('doc name redefined: ' + cmd)
                    else:
                        self.doc_name = cmd[:-1].strip()
                return

            # Context flag override: {+key1} means {key1:=1}, {/key2} means {key2:=0}
            # Also starts an unary chain: {+key1/key2/+key3*doc}
            # Other unary operators like `$:*` don't start a chain, as they yields
            # immediate values instead of preparing context.
            if cmd[0] == '+' or cmd[0] == '/':
                val = '1' if cmd[0] == '+' else '0'
                for i in range(2, len(cmd)):
                    if cmd[i] in all_punct:
                        key = cmd[1:i].strip()
                        ctx.setdefault_override(key, val)
                        if rest := cmd[i:]:
                            return handle_cmd(rest)
                        break
                else:
                    key = cmd[1:].strip()
                    ctx.setdefault_override(key, val)
                return

            for p, c in enumerate(cmd):
                if c in all_punct:
                    break

            # p is the first punctuation position, or len(cmd)
            if 0 < p < len(cmd):
                match cmd[p : p + 2]:
                    # Context override: {key:=value}
                    # a. Can never be overridden (except by markup buttons)
                    # b. Does not interrupt the natural order of flags detected in markup
                    # {key1:=key2=...=value} affects all keys
                    case ':=':
                        return self._do_assignments(
                            cmd, p, p + 2, ctx.setdefault_override
                        )

                    # Context set if empty: {key?=value}
                    # {key1?=key2=...=value} sets all keys to value if empty
                    case '?=':
                        return self._do_assignments(cmd, p, p + 2, ctx.setdefault)

                match cmd[p]:
                    # Context set: {key=value}
                    # {key1=key2=...=value} sets all keys to value
                    case '=':
                        return self._do_assignments(cmd, p, p + 1, ctx.__setitem__)

                    # Context inplace replacement: {key|pat1/rep1|pat2/rep2|...}
                    case '|':
                        key = cmd[:p].strip()
                        val = ctx.get(key)
                        if val is None:
                            return errors.append('undefined: ' + cmd)

                        parts = cmd[p + 1 :].split('|')
                        if not parts:
                            return errors.append('invalid replacements: ' + cmd)

                        r = str(val)
                        for part in parts:
                            try:
                                pat, rep = part.split('/', 1)
                            except ValueError:
                                errors.append('invalid replacement: ' + part)
                            else:
                                r = r.replace(pat.strip(), rep.strip())

                        ctx[key] = r
                        return

                    # Context swap: {key1^key2}
                    case '^':
                        key1 = cmd[:p].strip()
                        key2 = cmd[p + 1 :].strip()
                        val1 = ctx.get(key1)
                        val2 = ctx.get(key2)

                        self.set_ctx(key1, val2)
                        self.set_ctx(key2, val1)
                        return

            # Only the first expression is evaluated. Subsequent ones are discarded.
            if ret is None:
                ret = self.evaluate(cmd, in_directive=in_directive, as_str=True)
            else:
                errors.append('redundant expression: ' + cmd)

        # Find semicolons, but not inside quotes. Very trivial parsing.
        q0 = q1 = 1
        last = 0
        for p, c in enumerate(cmd):
            match c:
                case '\'':
                    q0 ^= 1
                case '\"':
                    q1 ^= 1
                case ';':
                    if q0 and q1:
                        handle_cmd(cmd[last:p].strip())
                        last = p + 1
        handle_cmd(cmd[last:].strip())

        return ret or ''

    def set_ctx(self, k: str, v: Value | None):
        if v is None:
            self.ctx.pop(k, None)
        else:
            self.ctx[k] = v

    def render(self, text: str) -> str:
        return reg.sub(self._repl, text).strip()

    def get_flag(self, key: str, default: bool = False) -> bool:
        v = self.ctx.get(key, default)
        return v and v != '0'


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


async def handle_render_doc(msg: Message):
    if not (text := msg.text) or not (text := text.strip()):
        return

    id = msg.message_id
    info = []

    ctx = RenderContext({}, this_doc=(id, text))
    rendered = ctx.render(text)

    if name := ctx.doc_name:
        preview_cache[id] = (name, rendered, ctx)
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
        reaction = (ReactionEmoji.RED_HEART, True)
    elif old_row := db.delete_doc(id):
        _report(info, 'deleted doc:', id, *old_row)
        reaction = ()
    else:
        return

    res = truncate_text(('\n' if len(info) > 2 else ' ').join(info))
    await asyncio.gather(
        do_notify(res, 'MarkdownV2', quiet=True),
        msg.set_reaction(*reaction),
    )
