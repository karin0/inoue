import re
import shlex
import asyncio

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
    CHAN_ID,
    TRUSTED_IDS,
)
from db import db
from context import Sender, ME, get_sender
from motto import greeting, hitokoto
from run import handle_cmd
from dispatch import (
    MessageArg,
    command,
    iter_commands,
    get_command_handler,
    dispatch_start,
)

try:
    from env import reply_usage
except ImportError:

    async def reply_usage(msg: Message, sender: Sender):
        text = rf'Hello, {escape(sender.name)}\!'
        await reply_text(msg, text, 'MarkdownV2')


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

    if handler := get_command_handler(cmd_name):
        with use_text_override(text):
            return await handler(update, ctx)

    return await handle_cmd(msg, content)


@command
async def handle_def(msg: Message, arg: MessageArg, bot: Bot):
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
        await set_commands(bot)
        await reply_text(msg, *pre_block(f'/{name} \u2192 {template}'))
    elif db.del_command(name):
        await set_commands(bot)
        await reply_text(msg, f'Deleted /{name}')
    else:
        await reply_text(msg, f'/{name} not found')


async def set_commands(bot: Bot):
    cmds = []
    public_cmds = []

    for name, (_, permissive) in iter_commands():
        cmds.append((name, name))
        if permissive or name == 'start':
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


@command
async def handle_greet(msg: Message, bot: Bot):
    await reply_text(msg, *stats(await bot.get_me()))


@command
async def handle_start(
    update: Update, ctx: ContextTypes.DEFAULT_TYPE, msg: Message, arg: MessageArg
):
    if not await dispatch_start(update, ctx, arg):
        sender = get_sender()
        if sender is None:
            raise RuntimeError('No sender')
        return await reply_usage(msg, sender)
