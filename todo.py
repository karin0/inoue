from typing import Any

from telegram import (
    Bot,
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.error import BadRequest

from db import db
from util import log, TODO_ID


def get_panel_id() -> int | None:
    val = db.get('todo_panel_id')
    return int(val) if val else None


def set_panel_id(msg_id: int | None):
    if msg_id is None:
        try:
            del db['todo_panel_id']
        except KeyError:
            pass
    else:
        db['todo_panel_id'] = str(msg_id)


async def switch_panel_id(bot: Bot, msg_id: int | None):
    panel_id = get_panel_id()
    set_panel_id(msg_id)
    if panel_id is not None:
        try:
            await bot.delete_message(chat_id=TODO_ID, message_id=panel_id)
        except BadRequest as e:
            log.error('Failed to delete old panel: %s', e)


def _build_panel() -> dict[str, Any] | None:
    tasks = db.get_todos()
    n = len(tasks)
    log.info('_build_panel: %d tasks', n)
    if not n:
        return

    text = f'{n} tasks:'
    rows = tuple(
        (InlineKeyboardButton(f'[{t_id}] {t_text}', callback_data=f'todo_{t_id}'),)
        for t_id, t_text in tasks
    )
    return {'text': text, 'reply_markup': InlineKeyboardMarkup(rows)}


async def handle_todo_msg(msg: Message):
    if not msg.text:
        return

    log.info('todo: %s', msg.text)

    for line in msg.text.splitlines():
        if text := line.strip():
            db.add_todo(text)

    if (kw := _build_panel()) is not None:
        sent = await msg.reply_text(**kw)
        await switch_panel_id(msg.get_bot(), sent.message_id)


async def handle_todo_callback(query: CallbackQuery, data: str, bot: Bot):
    _, task_id = data.split('_', 1)
    task_id = int(task_id)
    text = db.get_todo(task_id)
    if text is None:
        await query.answer('Task not found.', show_alert=True)
        return

    db.delete_todo(task_id)
    log.info('todo: done: %d: %s', task_id, text)

    if (kw := _build_panel()) is not None:
        try:
            await query.edit_message_text(**kw)
        except BadRequest as e:
            log.error('Failed to edit panel: %s', e)
            sent = await bot.send_message(TODO_ID, **kw)
            set_panel_id(sent.message_id)
    else:
        await switch_panel_id(bot, None)
