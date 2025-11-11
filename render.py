import re
import asyncio

from collections import UserDict

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
    MAX_TEXT_LENGTH,
    get_msg_arg,
    log,
    reply_text,
    shorten,
    truncate_text,
    escape,
    do_notify,
    get_msg_url,
)
from db import db

CALLBACK_SIGNS = '/+:'


def make_markup(
    text: str, ctx: dict[str], state: dict[str, bool] | None
) -> InlineKeyboardMarkup | None:
    # query data: ('/'|'+' <flag-key>)* ':' <text>
    # where '/' means 0, '+' means 1

    size = len(text) + 1
    if state:
        size += sum(len(k) for k in state.keys()) + len(state)

    if size > InlineKeyboardButton.MAX_CALLBACK_DATA:
        return None

    flags = {}
    for k, v in ctx.items():
        if v == '0':
            v = False
        elif v == '1':
            v = True
        else:
            continue

        if size + len(k) + 1 <= InlineKeyboardButton.MAX_CALLBACK_DATA:
            flags[k] = v

    if state:
        flags.update(state)

    if (doc_id := ctx.get(':db')) is None and not flags:
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

    return InlineKeyboardMarkup.from_row(row)


class OverriddenDict(UserDict):
    def __init__(self, overrides: dict[str, str]):
        super().__init__()
        self.overrides = overrides

    def __getitem__(self, key: str) -> str:
        if (v := self.overrides.get(key)) is not None:
            return v
        return super().__getitem__(key)

    # __setitem__ is not overridden, so we can still have the natural order
    # of the flags being inserted when iterated by `make_markup`.


def render_text(
    text: str, flags: dict[str, bool] | None = None
) -> tuple[str, InlineKeyboardMarkup | None, str | None]:
    if flags:
        ctx = OverriddenDict({k: '01'[v] for k, v in flags.items()})
    else:
        ctx = {}

    errors = []
    result = render(text, ctx, None, errors) or '[empty]'
    if errors:
        result += f'\n\n---\n\n' + '\n'.join(errors)

    result = truncate_text(result)
    parse_mode = None
    do_pre = ctx.get('_pre')
    if do_pre and do_pre != '0' and len(result) + 8 <= MAX_TEXT_LENGTH:
        result2 = f'```\n{escape(result)}\n```'
        if len(result2) <= MAX_TEXT_LENGTH:
            result = result2
            parse_mode = 'MarkdownV2'

    log.info('rendered %d -> %d', len(text), len(result))
    return truncate_text(result), make_markup(text, ctx, flags), parse_mode


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

    result, markup, parse_mode = render_text(text)
    return await reply_text(msg, result, reply_markup=markup, parse_mode=parse_mode)


async def handle_render_inline_query(query: InlineQuery, text: str):
    result, markup, parse_mode = render_text(text)
    await query.answer(
        [
            InlineQueryResultArticle(
                id='1',
                title=f'Render: {len(result)}',
                input_message_content=InputTextMessageContent(
                    result, parse_mode=parse_mode
                ),
                reply_markup=markup,
            )
        ]
    )


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

    result, markup, parse_mode = render_text(data[i + 1 :], flags)

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


reg = re.compile(r'{(.+?)}', re.DOTALL)


def render(
    text: str,
    ctx: dict[str],
    vis: set[str] | None,
    errors: list[str],
) -> str:
    def repl(m: re.Match) -> str:
        cmd = m[1].strip()
        if not cmd:
            return m[0]

        # Conditional: {cond ? true-directive : false-directive} (false part optional)
        if (p := cmd.find('?', 1)) >= 0:
            cond = cmd[:p].strip()
            if (t := cond.find('=', 1)) >= 0:
                key = cond[:t].strip()
                val = cond[t + 1 :].strip()
                test = str(ctx.get(key)) == val
            else:
                val = ctx.get(cond)
                test = val and val != '0'

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
            cmd = cmd.strip()

            # Empty directive
            if not cmd:
                continue

            # Literal: {`raw}
            elif cmd[0] == '`':
                return cmd[1:]

            # Db name set: {name:}
            elif cmd[-1] == ':':
                pass

            # Context set: {key=value}
            elif (p := cmd.find('=', 1)) >= 0:
                ctx[cmd[:p].strip()] = cmd[p + 1 :].strip()

            # Context swap: {key1^key2}
            elif (p := cmd.find('^', 1)) >= 0:
                key1 = cmd[:p].strip()
                key2 = cmd[p + 1 :].strip()
                val1 = ctx.get(key1)
                val2 = ctx.get(key2)

                def make(k, v):
                    if v is None:
                        if k in ctx:
                            del ctx[k]
                    else:
                        ctx[k] = v

                make(key1, val2)
                make(key2, val1)

            # Db doc get: {:name}
            elif cmd[0] == ':':
                key = cmd[1:].strip()

                nonlocal vis
                if vis is None:
                    vis = set()
                elif key in vis:
                    errors.append('circular: ' + cmd)
                    return ''

                if doc_row := db.get_doc(key):
                    doc_id, doc = doc_row
                else:
                    ctx.setdefault(':db', '')
                    errors.append('undefined: ' + cmd)
                    return ''

                if not ctx.get(':db'):
                    ctx[':db'] = doc_id

                vis.add(key)
                r = render(doc, ctx, vis, errors)
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

    return reg.sub(repl, text).strip()


reg_name = re.compile(r'{([\w\-]+?):}')


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


async def handle_doc(msg: Message):
    if not (text := msg.text) or not (text := text.strip()):
        return

    id = msg.message_id
    info = []

    if (m := reg_name.search(text)) and (name := m[1].strip()):
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

    await asyncio.gather(
        do_notify(
            truncate_text(('\n' if len(info) > 2 else ' ').join(info)),
            parse_mode='MarkdownV2',
        ),
        msg.set_reaction(*reaction),
    )
