import asyncio
import re
import shlex

from typing import TYPE_CHECKING

from telegram import (
    Bot,
    BotCommandScopeChat,
    BotCommandScopeChatAdministrators,
    BotCommandScopeDefault,
    User,
)

from bot import MessageArg, Responder, command, commands, escape

from .ctx import get_context
from .db import db
from .env import CHAN_ID, ME, TRUSTED_IDS
from .log import log
from .motto import greeting, hitokoto
from .run import handle_cmd
from .text import pre_block, pre_block_raw

if TYPE_CHECKING:
    from collections.abc import Awaitable

try:
    from conf import reply_usage  # pyright: ignore[reportMissingImports]
except ImportError as e:
    log.info('Using default reply_usage: %s', e)

    def reply_usage(rs: Responder) -> Awaitable:
        return rs.reply_cached(f'Hello, {get_context().sender_name}!')


if TYPE_CHECKING:

    def reply_usage(rs: Responder) -> Awaitable: ...


REG_TEMPLATE_ARG = re.compile(r'\$(\*|\d+)')


def expand_template(template: str, args_str: str) -> str:
    args = None

    def replacer(m):
        nonlocal args
        if (token := m.group(1)) == '*':
            return args_str
        idx = int(token) - 1
        if args is None:
            args = shlex.split(args_str)
        return args[idx] if 0 <= idx < len(args) else ''

    return REG_TEMPLATE_ARG.sub(replacer, template)


# Provided `rs.get_text()` must start with '/'.
def dispatch_cmd(rs: Responder, depth: int = 0) -> Awaitable:
    if depth > 10:
        return rs.reply_cached('Too many levels of command expansion.')

    if template := db.get_command(rs.get_cmd()):
        expanded = expand_template(template, rs.get_arg())
        if expanded.startswith('/'):
            rs.set_text(expanded)
            return dispatch_cmd(rs, depth + 1)
        raise RuntimeError(f'Bad command expansion: {expanded}')

    if handler := rs.get_route():
        return handler(rs)

    return handle_cmd(rs, rs.get_text()[1:])


@command
async def handle_def(rs: Responder, arg: MessageArg, bot: Bot):
    if not arg:
        cmds = []
        for name in db.iter_commands():
            cmd = db.get_command(name)
            cmds.append(f'/{name} \u2192 {cmd}')
        if not cmds:
            return await rs.reply_cached('No custom commands defined.')
        return await rs.reply_cached(*pre_block('\n'.join(cmds)))

    parts = arg.split(None, 1)
    name = parts[0]
    template = parts[1] if len(parts) > 1 else ''

    if template:
        db.set_command(name, template)
        await set_commands(bot)
        await rs.reply_cached(*pre_block(f'/{name} \u2192 {template}'))
    elif db.del_command(name):
        await set_commands(bot)
        await rs.reply_cached(f'Deleted /{name}')
    else:
        await rs.reply_cached(f'/{name} not found')


def set_commands(bot: Bot):
    cmds = []
    public_cmds = []

    for name, route in commands.items():
        cmds.append((name, name))
        if route.public or name == 'start':
            public_cmds.append((name, name))

    cmds.extend((name, name) for name in db.iter_commands())

    return asyncio.gather(
        bot.set_my_commands(public_cmds, BotCommandScopeDefault()),
        *(
            (
                bot.set_my_commands(cmds, BotCommandScopeChat(chat_id))
                if chat_id > 0
                else bot.set_my_commands(cmds, BotCommandScopeChatAdministrators(chat_id))
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
    text = (
        f'{escape(greeting())}\n{escape(hitokoto())}\n{pre_block_raw(f"{info}\n{kv}")}'
        f'{pre_block_raw(str(me))}'
    )
    return text, 'MarkdownV2'


@command
async def handle_greet(rs: Responder, bot: Bot):
    await rs.reply_cached(*stats(await bot.get_me()))


@command
def handle_start(rs: Responder, arg: MessageArg):
    if (coro := rs.dispatch_start()) is not None:
        return coro

    return reply_usage(rs)
