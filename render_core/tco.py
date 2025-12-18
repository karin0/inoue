from enum import Enum, auto
from typing import Any, Final, Literal
from lark import Tree


# TCO, but more like a signal to *break* from the current execution and "execve"
# into another block.
class _TCO(Enum):
    Tco = auto()


type TCO = Literal[_TCO.Tco]

type MaybeTCO = TCO | None

Tco: Final = _TCO.Tco

type TCOContext = tuple[Tree, str, dict[str, Any] | None]
