import re
import atexit
import asyncio
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
from telegram.helpers import escape_markdown
from telegram.constants import ReactionEmoji

from util import log, get_arg, shorten, truncate_text, do_notify, get_msg_url

db = None


def init_render(file):
    global db
    db = dbm.sqlite3.open(file, flag='c')
    atexit.register(close_render)
    check_integrity()


def close_render():
    global db
    if db is not None:
        db.close()
        db = None
        atexit.unregister(close_render)


def make_markup(
    text: str, ctx: dict[str], state: dict[str, bool] | None
) -> InlineKeyboardMarkup | None:
    # query data: ('-'|'+' <flag-key>)* ':' <text>
    # where '-' means 0, '+' means 1

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
            data = ''.join(('-+'[v] + k) for k, v in state.items()) + ':' + text
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
) -> tuple[str, InlineKeyboardMarkup | None]:
    if flags:
        ctx = OverriddenDict({k: '01'[v] for k, v in flags.items()})
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
    arg = get_arg()

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
                    ctx.setdefault(':db', '')
                    errors.append('undefined: ' + cmd)
                    return ''

                if not ctx.get(':db'):
                    ctx[':db'] = db['i:' + key].decode('utf-8')

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


def escape(s: str) -> str:
    return '\(`' + escape_markdown(s, version=2) + '`\)'


async def handle_doc(msg: Message):
    if not (text := msg.text) or not (text := text.strip()):
        return

    id = str(msg.message_id)
    n_key = 'n:' + id
    info = None
    reaction = None

    if (m := reg_name.search(text)) and (name := m[1].strip()):
        short = shorten(text)

        if old_name := db.get(n_key):
            if (old_name := old_name.decode('utf-8')) != name:
                del db['i:' + old_name]
                old_text = db.pop(old_name)
                db[n_key] = name

                log.info('renamed doc %s: %s -> %s (%s)', id, old_name, name, short)
                old_text = shorten(old_text.decode('utf-8'))
                info = rf'renamed doc {id}: {old_name} {escape(old_text)} \-\> [{name} {escape(short)}]({get_msg_url(id)})'
            else:
                log.info('updated doc %s: %s (%s)', id, name, short)
                info = f'updated doc [{id}: {name} {escape(short)}]({get_msg_url(id)})'
        else:
            db[n_key] = name

            log.info('new doc %s: %s (%s)', id, name, short)
            info = f'new doc [{id}: {name} {escape(short)}]({get_msg_url(id)})'

        if old_id := db.get(i_key := 'i:' + name):
            if (old_id := old_id.decode('utf-8')) != id:
                del db['n:' + old_id]
                db[i_key] = id

                log.info('unlinked doc %s: %s -> %s', name, old_id, id)

                old_text = shorten(db.pop(name).decode('utf-8'))
                info2 = rf'unlinked doc {name}: [{old_id} {escape(old_text)}]({get_msg_url(old_id)}) \-\> [{id}]({get_msg_url(id)})'
                info = (info + '\n' + info2) if info else info2
        else:
            db[i_key] = id

        db[name] = text

        reaction = (ReactionEmoji.RED_HEART, True)

    elif old_name := db.get(n_key):
        del db[n_key], db[b'i:' + old_name]
        old_text = db.pop(old_name)

        old_name = old_name.decode('utf-8')
        old_text = shorten(old_text.decode('utf-8'))

        log.warning('deleted doc %s: %s (%s)', id, old_name, old_text)
        info = f'deleted doc [{id}]({get_msg_url(id)}): {old_name} {escape(old_text)}'

        reaction = ()

    if reaction is not None:
        await asyncio.gather(
            do_notify(info, parse_mode='MarkdownV2'), msg.set_reaction(*reaction)
        )


def check_integrity(check: bool = True, fix: bool = False):
    for k in tuple(db.keys()):
        v = db[k]
        if k.startswith(b'n:'):
            log.debug('index: %s %s', k, v)
            id = k[2:]
            name = v
            i_key = b'i:' + name

            if fix:
                if (old_id := db.get(i_key)) == id:
                    pass
                elif old_id is None or (old_id := int(old_id.decode('utf-8'))) < int(
                    id.decode('utf-8')
                ):
                    log.error('inconsistent index: %s = %s -> %s', i_key, old_id, id)
                    db[i_key] = id
                elif old_id > id:
                    log.error(
                        'inconsistent index: %s = %s > %s (del)', i_key, old_id, id
                    )
                    del db[k]
                    continue

                if name not in db:
                    log.error('inconsistent index: no doc for %s %s (del)', id, name)
                    del db[k], db[i_key]

            elif check:
                assert db[i_key] == id
                assert name in db

        elif k.startswith(b'i:'):
            log.debug('index: %s %s', k, v)
            if check:
                assert db[b'n:' + v] == k[2:]
        else:
            log.debug('doc: %s %s', k, shorten(v.decode('utf-8')))
            if check:
                assert db[b'n:' + db[b'i:' + k]] == k
