'''Stand-in for `bot.app`, which builds a Telegram client at import time.'''

import asyncio

from unittest.mock import MagicMock

app = MagicMock(name='app')
bot = MagicMock(name='bot')
create_task = asyncio.create_task
