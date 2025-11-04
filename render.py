import re
import atexit
import dbm.sqlite3
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

from util import get_arg, shorten, truncate_text, log, log2

db = None


def init_render(file):
    global db
    db = dbm.sqlite3.open(file, flag='c')
    atexit.register(close_render)


def close_render():
    global db
    if db is not None:
        db.close()
        db = None
        atexit.unregister(close_render)


def make_markup(
    text: str, ctx: dict[str], prev_flags: dict[str, bool] | None
) -> InlineKeyboardMarkup | None:
    # query data: ('-'|'+' <flag-key>)* ':' <text>
    # where '-' means 0, '+' means 1

    size = len(text) + 1
    if prev_flags:
        size += sum(len(k) for k in prev_flags.keys()) + len(prev_flags)

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

    if prev_flags:
        flags.update(prev_flags)

    if not (':db' in ctx or flags):
        return None

    if flags:
        row = [
            InlineKeyboardButton(text='⏪ ' + shorten(text), callback_data=':' + text)
        ]
        for k, v in flags.items():
            hit = False
            data = (
                ''.join(
                    ('+' if (((hit := True) and not vv) if k == kk else vv) else '-')
                    + kk
                    for kk, vv in prev_flags.items()
                )
                if prev_flags
                else ''
            )
            data += ((('-' if v else '+') + k + ':') if not hit else ':') + text
            row.append(
                InlineKeyboardButton(
                    text=k + '=' + ('1' if v else '0'), callback_data=data
                )
            )
        row.append(
            InlineKeyboardButton(
                text='🔄',
                callback_data=(
                    ''.join(('+' if v else '-') + k for k, v in prev_flags.items())
                    + ':'
                    + text
                ),
            )
        )
    else:
        row = [
            InlineKeyboardButton(text='🔄 ' + shorten(text), callback_data=':' + text)
        ]

    return InlineKeyboardMarkup.from_row(row)


class OverriddenDict(UserDict):
    def __init__(self, overrides: dict[str, str]):
        super().__init__()
        self.overrides = overrides

    def __getitem__(self, key: str) -> str:
        if key in self.overrides:
            return self.overrides[key]
        return super().__getitem__(key)


def render_text(
    text: str, flags: dict[str, bool] | None = None
) -> tuple[str, InlineKeyboardMarkup | None]:
    if flags:
        ctx = OverriddenDict({k: ('1' if v else '0') for k, v in flags.items()})
    else:
        ctx = {}

    errors = []
    result = render(text, ctx, None, errors) or '[empty]'
    if errors:
        result += f'\n\n---\n\n' + '\n'.join(errors)

    log.info('rendered %d -> %d', len(text), len(result))
    return truncate_text(result), make_markup(text, ctx, flags)


async def handle_render(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message or update.edited_message
    target = msg.reply_to_message
    arg = get_arg(update)

    if target:
        text = (target.text or target.caption).strip()
        if not text:
            return await msg.reply_text('No text to render.', do_quote=True)

        if arg:
            # Include context set by arg
            text = arg + '\n' + text
    elif arg:
        text = arg
    else:
        return await msg.reply_text(
            'Specify text or reply to a message to render.', do_quote=True
        )

    result, markup = render_text(text)
    return await msg.reply_text(result, reply_markup=markup, do_quote=True)


async def handle_render_inline_query(query: InlineQuery, text: str):
    result, markup = render_text(text)
    await query.answer(
        [
            InlineQueryResultArticle(
                id='1',
                title=f'Render: {len(result)}',
                input_message_content=InputTextMessageContent(result),
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
        while j < len(data) and data[j] not in '-+:':
            j += 1
        key = data[i:j]
        flags[key] = sign == '+'
        i = j

    result, markup = render_text(data[i + 1 :], flags)

    try:
        if update.callback_query.inline_message_id:
            await ctx.bot.edit_message_text(
                text=result,
                inline_message_id=update.callback_query.inline_message_id,
                reply_markup=markup,
            )
        else:
            await ctx.bot.edit_message_text(
                text=result,
                chat_id=update.effective_chat.id,
                message_id=update.callback_query.message.message_id,
                reply_markup=markup,
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
        if (p := cmd.find('?')) != -1:
            cond = cmd[:p].strip()
            if (t := cond.find('=')) != -1:
                key = cond[:t].strip()
                val = cond[t + 1 :].strip()
                test = str(ctx.get(key)) == val
            else:
                val = ctx.get(cond)
                test = val and val != '0'

            # Prefer doc expansion for ?:
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
            elif (p := cmd.find('=')) != -1:
                ctx[cmd[:p].strip()] = cmd[p + 1 :].strip()

            # Context swap: {key1^key2}
            elif (p := cmd.find('^')) != -1:
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
                ctx[':db'] = True
                key = cmd[1:].strip()

                nonlocal vis
                if vis is None:
                    vis = set()
                elif key in vis:
                    errors.append('circular: ' + cmd)
                    return ''

                try:
                    doc = db[key].decode('utf-8')
                except KeyError:
                    errors.append('undefined: ' + cmd)
                    return ''

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


reg_name = re.compile(r'{(.+?):}')


async def handle_doc(msg: Message):
    text = msg.text
    if not text or not (text := text.strip()):
        return await msg.reply_text('No text to process.', do_quote=True)

    id = str(msg.message_id)
    name_key = 'n:' + id

    if (m := reg_name.search(text)) and (name := m[1].strip()):
        db[name] = text

        if (old_name := db.get(name_key)) and (
            old_name := old_name.decode('utf-8')
        ) != name:
            log2.info('rename doc %s -> %s %s %s', old_name, name, id, shorten(text))
            del db[old_name]
        else:
            log2.info('update doc %s %s %s', name, id, shorten(text))

        db[name_key] = name

        await msg.set_reaction(ReactionEmoji.RED_HEART, True)

    elif old_name := db.get(name_key):
        del db[old_name], db[name_key]
        log2.info('delete doc %s %s', old_name, id)

        await msg.set_reaction()
