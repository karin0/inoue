import re
import atexit
import dbm.sqlite3

from telegram import (
    Update,
    Message,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import ContextTypes
from telegram.error import BadRequest

from util import get_arg, shorten, log

db = None


def init_render(file):
    global db
    db = dbm.sqlite3.open(file, flag='c')
    atexit.register(db.close)


def make_markup(data: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup.from_button(
        InlineKeyboardButton(text='Refresh', callback_data=data)
    )


def render_text(text: str, ctx: dict[str]) -> str:
    errors = []
    result = render(text, ctx, None, errors) or '[empty]'
    if errors:
        result += f'\n\n---\n\n' + '\n'.join(errors)

    log.info('rendered %d -> %d', len(text), len(result))
    return result


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

    ctx = {}
    result = render_text(text, ctx)

    if ':db' in ctx and (
        InlineKeyboardButton.MIN_CALLBACK_DATA
        <= len(text) + 1
        <= InlineKeyboardButton.MAX_CALLBACK_DATA
    ):
        markup = make_markup(':' + text)
    else:
        markup = None

    return await msg.reply_text(result, reply_markup=markup, do_quote=True)


async def handle_render_callback(
    update: Update, bot_ctx: ContextTypes.DEFAULT_TYPE, data: str
):
    result = render_text(data[1:], {})

    try:
        await bot_ctx.bot.edit_message_text(
            text=result,
            chat_id=update.effective_chat.id,
            message_id=update.callback_query.message.message_id,
            reply_markup=make_markup(data),
        )
    except BadRequest as e:
        if 'Message is not modified' not in str(e):
            raise


reg = re.compile(r'{(.+?)}', re.DOTALL)


def render(text: str, ctx: dict[str], vis_: set[str] | None, errors: list[str]) -> str:
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
                test = bool(ctx.get(cond))

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

            # Db name set: {name:}
            if cmd[-1] == ':':
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

                vis = vis_
                if vis is None:
                    vis = set()
                elif key in vis:
                    errors.append('circular: ' + key)
                    return ''

                try:
                    doc = db[key].decode('utf-8')
                except KeyError:
                    errors.append('undefined: ' + key)
                    return ''

                vis.add(key)
                r = render(doc, ctx, vis, errors)
                vis.remove(key)
                return r

            # Literal: {`raw}
            elif cmd[0] == '`':
                return cmd[1:]

            # Context get: {$key} or {key} (if no condition)
            elif cmd[0] == '$' or not cond:
                key = cmd.lstrip('$').strip()
                try:
                    return str(ctx[key])
                except KeyError:
                    ctx.setdefault(':undefined', []).append(key)
                    return ''

            # Literal: {raw} (if condition present)
            else:
                return cmd

        return ''

    return reg.sub(repl, text).strip()


reg_name = re.compile(r'{(.+?):}')


async def handle_doc(msg: Message):
    text = msg.text
    if not text:
        return await msg.reply_text('No text to process.', do_quote=True)

    text = text.strip()

    if m := reg_name.search(text):
        name = m[1].strip()
        id = str(msg.message_id)

        db[name] = text

        if (old_name := db.get('n:' + id)) and old_name.decode('utf-8') != name:
            del db[old_name]

        db['n:' + id] = name

        log.info('doc %s %s %s', name, id, shorten(text))
