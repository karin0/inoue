from . import bot as _bot_util

from .env import *
from .ctx import *
from .log import *
from .text import *
from .bot import *


def init_util(b: Bot):
    _bot_util.bot = b
