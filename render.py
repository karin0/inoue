import re
import asyncio

from typing import Iterable
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

CALLBACK_SIGNS = '/+:'


class OverriddenDict(UserDict):
    def __init__(self, overrides: dict[str, str]):
        super().__init__()
        self.overrides = overrides

    def __getitem__(self, key: str) -> str:
        if (v := self.overrides.get(key)) is not None:
            return v
        return self.data[key]

    # For `get` method to work.
    def __contains__(self, key: str) -> bool:
        return key in self.overrides or key in self.data

    def setdefault(self, key: str, default: str):
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

    def setdefault_override(self, key: str, value: str) -> str:
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

    def finalize(self):
        # Keys from the underlying dict are yielded first, in their natural order.
        r = self.data
        r.update(self.overrides)
        return r.items()


def make_markup(
    text: str, ctx: 'RenderContext', state: dict[str, bool] | None
) -> InlineKeyboardMarkup | None:
    # query data: ('/'|'+' <flag-key>)* ':' <text>
    # where '/' means 0, '+' means 1

    size = len(text) + 1
    if state:
        size += sum(len(k.encode('utf-8')) for k in state.keys()) + len(state)

    if size > InlineKeyboardButton.MAX_CALLBACK_DATA:
        return None

    flags = {}
    for k, v in ctx.ctx.finalize():
        if v == '0':
            v = False
        elif v == '1':
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
            ('⏪ ' if state else '🔄 ') + shorten(text), callback_data=':' + text
        )
    ]

    if flags:
        if state is None:
            state = {}

        def push_button(name: str):
            data = (
                ''.join((CALLBACK_SIGNS[v] + k) for k, v in state.items()) + ':' + text
            )
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


def finalize_rendered_text(
    text: str,
    flags: dict[str, bool] | None,
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

    log.info('rendered %d -> %d', len(text), len(result))
    return truncate_text(result), parse_mode, make_markup(text, ctx, flags)


def render_text(
    text: str, flags: dict[str, bool] | None = None
) -> tuple[str, str | None, InlineKeyboardMarkup | None]:
    ctx = RenderContext({k: '01'[v] for k, v in flags.items()} if flags else {})
    return finalize_rendered_text(text, flags, ctx.render(text), ctx)


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

    return await reply_text(msg, *render_text(text))


preview_cache = {}


async def handle_render_group(msg: Message, origin_id: int):
    if cache := preview_cache.pop(origin_id, None):
        doc_name, args = cache
        log.info('Doc in group: %s -> %s %s', msg.id, origin_id, doc_name)
        await reply_text(msg, *args, allow_not_modified=True)
    else:
        log.warning('No doc cache for group: %s -> %s', msg.id, origin_id)

    if left := tuple((id, name) for id, (name, _) in preview_cache.items()):
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
    i = 0
    while i < len(data):
        sign = data[i]
        if sign == ':':
            break

        i += 1
        j = i
        while j < len(data) and data[j] not in CALLBACK_SIGNS:
            j += 1
        key = data[i:j]
        flags[key] = sign == CALLBACK_SIGNS[True]
        i = j

    result, parse_mode, markup = render_text(data[i + 1 :], flags)

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


def _flatten_keys(key1: str, parts: list[str]) -> Iterable[str]:
    yield key1
    for key in parts:
        yield key.strip()


def split_assignments(cmd: str, p1: int, p2: int) -> tuple[str, Iterable[str]]:
    key1 = cmd[:p1].strip()
    parts = cmd[p2:].split('=')
    val = parts.pop().strip()
    return val, _flatten_keys(key1, parts)


reg = re.compile(r'{(.+?)}|^([^\r\n]+?);$', re.DOTALL | re.MULTILINE)


class RenderContext:
    def __init__(self, overrides: dict[str]):
        self.ctx = OverriddenDict(overrides)
        self.vis: set[str] = set()
        self.errors: list[str] = []
        self.first_doc_id: int | None = None
        self.first_doc_id_is_implicit: bool = False
        self.doc_name = None

    def _repl(self, m: re.Match, top: bool) -> str:
        if not (cmd := m[1] or m[2]):
            return m[0]

        ctx = self.ctx
        errors = self.errors

        # Conditional: {cond ? true-directive : false-directive} (false part optional)
        # Omit `?=` which is handled inside directives and cannot occur in conditions
        if (p := cmd.find('?', 1)) >= 0 and p + 1 < len(cmd) and cmd[p + 1] != '=':
            cond = cmd[:p].strip()
            if not cond:
                errors.append('empty condition: ' + cmd)
                test = False
            elif cond[0] == cond[-1] == '\"':
                # Evaluate Python expression
                expr = cond[1:-1].strip()
                try:
                    v = simple_eval(expr, names=ctx.clone())
                except Exception as e:
                    errors.append(f'evaluation: {expr}: {type(e).__name__}: {e}')
                    test = False
                else:
                    test = bool(v)
                    log.info('evaluated condition: %s -> %s (%s)', expr, v, test)
            elif (t := cond.find('=', 1)) >= 0:
                key = cond[:t].strip()
                val = cond[t + 1 :].strip()
                test = str(ctx.get(key)) == val
            else:
                test = self.get_flag(cond)

            # Prefer doc expansion for ?:<doc>
            if (q := cmd.find(':', p + 2)) != -1:
                cmd = cmd[p + 1 : q].strip() if test else cmd[q + 1 :].strip()
            elif test:
                cmd = cmd[p + 1 :].strip()
            else:
                return ''
        else:
            cond = None

        for cmd in cmd.split(';'):
            cmd: str = cmd.strip()

            # Empty directive
            if not cmd:
                pass

            # Literal: {`raw}
            elif cmd[0] == '`':
                return cmd[1:]

            # Db name set: {name:}
            elif cmd[-1] == ':':
                if top:
                    if self.doc_name:
                        errors.append('doc name redefined: ' + cmd)
                    else:
                        self.doc_name = cmd[:-1].strip()

            # Context override: {key:=value}
            # a. Can never be overridden (except by markup buttons)
            # b. Does not interrupt the natural order of flags detected in markup
            # {key1:=key2=...=value} affects all keys
            elif (p := cmd.find(':=', 1)) >= 0:
                val, keys = split_assignments(cmd, p, p + 2)
                for key in keys:
                    ctx.setdefault_override(key, val)

            # Context set if empty: {key?=value}
            # {key1?=key2=...=value} sets all keys to value if empty
            elif (p := cmd.find('?=', 1)) >= 0:
                val, keys = split_assignments(cmd, p, p + 2)
                for key in keys:
                    ctx.setdefault(key, val)

            # Context set: {key=value}
            # {key1=key2=...=value} sets all keys to value
            elif (p := cmd.find('=', 1)) >= 0:
                val, keys = split_assignments(cmd, p, p + 1)
                for key in keys:
                    ctx[key] = val

            # Context inplace replacement: {key|pat1/rep1|pat2/rep2|...}
            elif (p := cmd.find('|', 1)) >= 0:
                key = cmd[:p].strip()
                val = ctx.get(key)
                if val is None:
                    errors.append('undefined: ' + cmd)
                    continue

                parts = cmd[p + 1 :].split('|')
                if not parts:
                    errors.append('invalid replacements: ' + cmd)
                    continue

                r = str(val)
                for part in parts:
                    try:
                        pat, rep = part.split('/', 1)
                    except ValueError:
                        errors.append('invalid replacement: ' + part)
                        continue
                    r = r.replace(pat.strip(), rep.strip())
                ctx[key] = r

            # Context swap: {key1^key2}
            elif (p := cmd.find('^', 1)) >= 0:
                key1 = cmd[:p].strip()
                key2 = cmd[p + 1 :].strip()
                val1 = ctx.get(key1)
                val2 = ctx.get(key2)

                self._ctx_set(key1, val2)
                self._ctx_set(key2, val1)

            # Db doc get: {:name}
            # Db doc implicit get: {*name} (can be overridden as self.first_doc_id)
            elif cmd[0] == ':' or cmd[0] == '*':
                key = cmd[1:].strip()
                vis = self.vis

                if key in vis:
                    errors.append('circular: ' + cmd)
                    return ''

                if doc_row := db.get_doc(key):
                    doc_id, doc = doc_row
                else:
                    errors.append('undefined: ' + cmd)
                    return ''

                if self.first_doc_id is None or self.first_doc_id_is_implicit:
                    self.first_doc_id = doc_id
                    self.first_doc_id_is_implicit = cmd[0] == '*'

                vis.add(key)
                r = self.render(doc, top=False)
                vis.remove(key)
                return r

            # Context get: {$key} or {key} (w/o condition)
            elif cmd[0] == '$' or not cond:
                key = cmd.lstrip('$').strip()

                if (val := ctx.get(key)) is None:
                    errors.append('undefined: ' + cmd)
                    return ''

                return str(val)

            # Literal: {raw} (when condition)
            else:
                return cmd

        return ''

    def _ctx_set(self, k, v):
        ctx = self.ctx
        if v is None:
            if k in ctx:
                del ctx[k]
        else:
            ctx[k] = v

    def render(self, text: str, top: bool = True) -> str:
        return reg.sub(lambda m: self._repl(m, top), text).strip()

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

    ctx = RenderContext({})
    rendered = ctx.render(text)

    if name := ctx.doc_name:
        args = finalize_rendered_text(f'*{name};', None, rendered, ctx)
        preview_cache[id] = (name, args)

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
