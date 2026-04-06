from typing import Iterator, Mapping
from collections import UserDict, OrderedDict
from collections.abc import ItemsView, MutableMapping

from db import db
from util import log
from render_core import Context, Value, is_value_type

LRU_CAPACITY = 128

type Items = ItemsView[str, Value]


def encode_value(val: Value) -> str:
    if isinstance(val, bool):
        return 'b' if val else 'B'
    if isinstance(val, int):
        return ('i' + str(val)) if val >= 0 else ('I' + str(-val))
    if isinstance(val, float):
        return ('f' + str(val)) if val >= 0 else ('F' + str(-val))
    return 's' + str(val)


def decode_value(s: str) -> Value:
    val = s[1:]
    match s[0]:
        case 'b':
            return True
        case 'B':
            return False
        case 'i':
            return int(val)
        case 'I':
            return -int(val)
        case 'f':
            return float(val)
        case 'F':
            return -float(val)
        case 's':
            return val
        case _:
            raise ValueError('bad encoded value: ' + s)


class LRUDict(MutableMapping[str, Value]):
    def __init__(self):
        self._data = OrderedDict()

    def __contains__(self, key) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Value:
        self._data.move_to_end(key)
        return self._data[key]

    def __setitem__(self, key: str, value: Value) -> None:
        log.debug('PM set: %s = %r', key, value)
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > LRU_CAPACITY:
            self._data.popitem(last=False)

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        return self._data.clear()

    def items(self) -> Items:
        return self._data.items()


class PersistentStorage(MutableMapping[str, Value]):
    def __init__(self) -> None:
        super().__init__()
        self._cache = LRUDict()

    def __getitem__(self, key: str) -> Value:
        if key in self._cache:
            return self._cache[key]
        value = decode_value(db['pm-' + key])
        self._cache[key] = value
        return value

    def __setitem__(self, key: str, value: Value) -> None:
        db['pm-' + key] = encode_value(value)
        self._cache[key] = value

    def __delitem__(self, key: str) -> None:
        del db['pm-' + key]
        del self._cache[key]

    def __iter__(self) -> Iterator[str]:
        return iter(k[3:] for k in db.iter_prefix('pm-'))

    def __len__(self) -> int:
        return db.count_prefix('pm-')


# Static context variables are persisted across different `RenderInterpreter` instances.
persisted = PersistentStorage()
PM_PREFIX = 'pm.'


def get_pm_key(key: str) -> str | None:
    if len(pm_key := key.removeprefix(PM_PREFIX)) != len(key):
        return pm_key


class OverriddenDict(UserDict[str, Value], Context):
    def __init__(self, data: Mapping[str, Value], overrides: dict[str, Value]):
        for val in data.values():
            if not is_value_type(val):
                raise TypeError(f'bad data value type: {type(val)}: {val}')
        super().__init__(data)

        for val in overrides.values():
            if not is_value_type(val):
                raise TypeError(f'bad override value type: {type(val)}: {val}')
        self.overrides = overrides

    def __getitem__(self, key: str) -> Value:
        if (v := self.overrides.get(key)) is not None:
            return v
        if pm_key := get_pm_key(key):
            return persisted[pm_key]
        return self.data[key]

    # For `get` method to work.
    def __contains__(self, key: str) -> bool:  # type: ignore[override]
        if key in self.overrides:
            return True
        if pm_key := get_pm_key(key):
            return pm_key in persisted
        return key in self.data

    # For `set_or_del_raw` to work.
    def __delitem__(self, key: str):
        if pm_key := get_pm_key(key):
            if key not in self.overrides:
                del persisted[pm_key]
            return

        if key not in self.overrides:
            del self.data[key]

    # For `=` operator.
    def __setitem__(self, key: str, val: Value):
        if pm_key := get_pm_key(key):
            if key not in self.overrides:
                persisted[pm_key] = val
            return

        # We always set the value in the underlying dict, even if the value
        # is overridden, so the natural order of keys is preserved as defined by
        # documents and won't change after markup buttons are activated.
        self.data[key] = val

    # For `:=` operator.
    # This only affects the `overrides` dict, which has higher priority than
    # the underlying dict by their *values*, but are placed after those touched
    # by `=` and `?=` in the natural order of *keys*.
    # However, the existing `overrides` keys are frozen and can NEVER be modified
    # or removed, even by `:=` itself.
    def setitem_with(self, key: str, val: Value, op: str) -> None:
        self.overrides.setdefault(key, val)

    def _compact(self):
        return dict(self.data, **self.overrides)

    def __iter__(self):
        # Keys from the underlying dict are yielded first, in their natural order.
        # Technically the side effects of this method shouldn't affect our behavior,
        # since existing overrides can never be removed or modified.
        return iter(self._compact())

    def __len__(self) -> int:
        return len(self._compact())

    def items(self) -> Items:
        return self._compact().items()
