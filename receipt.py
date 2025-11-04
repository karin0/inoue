import functools
from math import floor
from typing import Hashable
from collections import defaultdict

from arithmetic_eval import evaluate


def calc(s: str) -> float:
    return evaluate(s, {'x': 1.1})


def render_sign(v: float | int) -> str:
    if isinstance(v, int):
        return f'{v:+}'
    return f'{v:+.3}'


def render_pair(x, y) -> str:
    if y == 0:
        return str(x)
    return f'{x} ({render_sign(y)})'


class Value:
    def __init__(self, raw: float, kind: Hashable = None):
        self.raw = raw
        self.cost = round(raw)
        self.kind = kind  # Total cost is rounded for each group by kind

    @property
    def adj(self):
        return self.cost - self.raw

    def __str__(self):
        return render_pair(self.cost, self.adj)

    __repr__ = __str__


def alter(items: list[Value], real_tot: int):
    d = sum(item.cost for item in items) - real_tot
    if not d:
        return

    t = sorted(items, key=lambda x: x.adj, reverse=d > 0)
    off = -1 if d > 0 else 1
    d = abs(d)
    assert d <= len(t)

    for x in t[:d]:
        x.cost += off

    assert real_tot == sum(item.cost for item in items)


def render_receipt(text: str):
    the_real_tot: int | None = None
    tax: float = 1.08
    items: list[Value] = []
    alts: list[tuple[float, int]] = []

    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) == 2 and not parts[0][0].isdigit():
            cmd, arg = parts
            if cmd == '=':
                if the_real_tot is not None:
                    raise SyntaxError('Duplicate "="')
                the_real_tot = int(arg)
            elif cmd.startswith('t'):
                tax = calc(arg)
            elif cmd.startswith('a'):
                # Find the precision of the argument
                p = arg.find('.')
                if p >= 0 and all(i == p or c.isdigit() for i, c in enumerate(arg)):
                    base = 10 ** (len(arg) - p - 1)
                    alts.append((int(calc(arg) * base), base))
                else:
                    alts.append((int(calc(arg)), 1))
            else:
                raise SyntaxError(f'Unknown command: {cmd}')
            continue

        if len(parts) == 1:
            items.append(Value(calc(parts[0]) * tax, tax))
        else:
            parts = tuple(calc(p) for p in parts)
            items.append(Value(functools.reduce(lambda x, y: x * y, parts), parts[1]))

    if not items:
        raise ValueError('No values provided')

    tot_dict = defaultdict(float)
    for item in items:
        tot_dict[item.kind] += item.raw
    real_tot = sum(floor(v) for v in tot_dict.values())

    alter(items, real_tot)
    tots = [render_pair(real_tot, real_tot - sum(item.raw for item in items))]

    dss = [items]
    if real_tot:
        for tot, base in alts:
            alt_items = [Value(x.cost * tot / real_tot) for x in items]
            alter(alt_items, tot)
            if base != 1:
                tot /= base
                for x in alt_items:
                    x.cost /= base
                    x.raw /= base
            tots.append(f'{tot} (x{tot / real_tot:.3f})')
            dss.append(alt_items)

    res = [' |\t'.join(str(x) for x in row) for row in zip(*dss)]
    res.append('= ' + ' |\t'.join(tots))

    if the_real_tot is not None and the_real_tot != real_tot:
        res.append(f'⚠️ {the_real_tot} (expected) != {real_tot}')

    return '\n'.join(res)
