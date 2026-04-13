import re
import shlex
import asyncio
from typing import Callable, Sequence

from telegram.ext import ContextTypes
from telegram import (
    Message,
    User,
    Update,
    Bot,
    BotCommandScopeDefault,
    BotCommandScopeChat,
    BotCommandScopeChatAdministrators,
)

from util import (
    log,
    escape,
    pre_block,
    pre_block_raw,
    reply_text,
    use_text_override,
    get_msg_arg,
    ME,
    CHAN_ID,
    TRUSTED_IDS,
)
from db import db
from render import handle_render
from motto import greeting, hitokoto
from rg import handle_rg, handle_rg_start
from misc import handle_sort, handle_fetch
from run import handle_run, handle_cmd, handle_update
from voice import handle_voice
from media import *

REG_TEMPLATE_ARG = re.compile(r'\$(\*|\d+)')

commands: dict[str, tuple[Callable, bool]]


def expand_template(template: str, args_str: str) -> str:
    args = shlex.split(args_str) if args_str else []

    def replacer(m):
        token = m.group(1)
        if token == '*':
            return args_str
        idx = int(token) - 1
        return args[idx] if 0 <= idx < len(args) else ''

    return REG_TEMPLATE_ARG.sub(replacer, template)


# Provided `text` must start with '/'.
async def dispatch_cmd(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    msg: Message,
    text: str,
    depth: int = 0,
):
    if depth > 10:
        return await reply_text(msg, 'Too many levels of command expansion.')

    content = text[1:].strip()
    parts = content.split(None, 1)
    cmd_name = parts[0].split('@')[0]
    cmd_args = parts[1] if len(parts) > 1 else ''

    if template := db.get_command(cmd_name):
        expanded = expand_template(template, cmd_args)
        if expanded.startswith('/'):
            return await dispatch_cmd(update, ctx, msg, expanded, depth + 1)
        return await handle_cmd(msg, expanded)

    if handler := commands.get(cmd_name):
        with use_text_override(text):
            return await handler[0](update, ctx)

    return await handle_cmd(msg, content)


async def handle_def(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg, arg = get_msg_arg(update)
    if not arg:
        cmds = []
        for name in db.iter_commands():
            cmd = db.get_command(name)
            cmds.append(f'/{name} \u2192 {cmd}')
        if not cmds:
            return await reply_text(msg, 'No custom commands defined.')
        return await reply_text(msg, *pre_block('\n'.join(cmds)))

    parts = arg.split(None, 1)
    name = parts[0]
    template = parts[1] if len(parts) > 1 else ''

    if template:
        db.set_command(name, template)
        await set_commands(ctx.bot)
        await reply_text(msg, *pre_block(f'/{name} \u2192 {template}'))
    else:
        if db.get_command(name):
            db.set_command(name, '')
            await set_commands(ctx.bot)
            await reply_text(msg, f'Deleted /{name}')
        else:
            await reply_text(msg, f'/{name} not found')


async def set_commands(bot: Bot):
    cmds = []
    public_cmds = []

    for name, (func, permissive) in commands.items():
        cmds.append((name, name))
        if permissive or func is handle_start:
            public_cmds.append((name, name))

    for name in db.iter_commands():
        cmds.append((name, name))

    await asyncio.gather(
        bot.set_my_commands(public_cmds, BotCommandScopeDefault()),
        *(
            (
                bot.set_my_commands(cmds, BotCommandScopeChat(chat_id))
                if chat_id > 0
                else bot.set_my_commands(
                    cmds, BotCommandScopeChatAdministrators(chat_id)
                )
            )
            for chat_id in TRUSTED_IDS
            if chat_id != CHAN_ID
        ),
    )


def stats(me: User, header: str = ME) -> tuple[str, str]:
    info = f'{header}: {db.summary()}'
    kv = []
    tot = 0
    for prefix in ('r', 'pm', 'u', 'c', 'G%'):
        cnt = db.count_prefix(prefix + '-')
        kv.append(f'{prefix}: {cnt}')
        tot += cnt
    kv.append(f'tot: {tot}')
    kv = ', '.join(kv)
    log.info('%s (%s)', info, kv)
    text = f'{escape(greeting())}\n{escape(hitokoto())}\n{pre_block_raw(f"{info}\n{kv}")}{pre_block_raw(str(me))}'
    return text, 'MarkdownV2'


async def handle_greet(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await reply_text(update, *stats(await ctx.bot.get_me()))


def parse_media(arg: str) -> tuple[int, int] | None:
    parts = arg.split('_', 2)
    if len(parts) != 3:
        return None

    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return None


async def handle_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg, arg = get_msg_arg(update)
    if arg:
        if arg.startswith('rg_'):
            return await handle_rg_start(msg, arg)

        elif arg.startswith('play_'):
            if (parsed := parse_media(arg)) is None:
                return await reply_text(msg, 'Bad play link.')
            return await handle_play_media(msg, *parsed)

        elif arg.startswith('unsave_'):
            if (parsed := parse_media(arg)) is None:
                return await reply_text(msg, 'Bad unsave link.')
            return await handle_remove_media(msg, *parsed)

    return await reply_text(msg, *stats(await ctx.bot.get_me()))


handlers = (
    handle_start,
    handle_update,
    handle_rg,
    handle_run,
    handle_fetch,
    handle_greet,
    handle_def,
    handle_save,
    handle_playlist,
)

permissive_handlers = (
    handle_play,
    handle_render,
    handle_voice,
    handle_sort,
)


def to_commands(
    funcs: Sequence[Callable], permissive: bool
) -> dict[str, tuple[Callable, bool]]:
    return {f.__name__[f.__name__.index('_') + 1 :]: (f, permissive) for f in funcs}


commands = to_commands(handlers, False) | to_commands(permissive_handlers, True)
