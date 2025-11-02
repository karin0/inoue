import functools
from math import floor
from typing import Hashable
from collections import defaultdict


def calc(s: str) -> float:
    if s == 'x':
        return 1.1
    return eval(s)


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


def render_calc(text: str):
    the_real_tot: int | None = None
    tax: float = 1.08
    items: list[Value] = []
    alt_tots: list[float] = []
    alt_bases: list[int] = []

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
                if p >= 0:
                    base = 10 ** (len(arg) - p - 1)
                    alt_bases.append(base)
                    alt_tots.append(int(calc(arg) * base))
                else:
                    alt_bases.append(1)
                    alt_tots.append(calc(arg))
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

    if not real_tot:
        raise ValueError('You are free!')

    if the_real_tot is not None and the_real_tot != real_tot:
        raise ValueError(f'{the_real_tot} != {real_tot}')

    alter(items, real_tot)
    results = [render_pair(real_tot, real_tot - sum(item.raw for item in items))]

    dss = [items]
    for tot, base in zip(alt_tots, alt_bases):
        alt_items = [Value(x.cost * tot / real_tot) for x in items]
        alter(alt_items, tot)
        if base != 1:
            tot /= base
            for x in alt_items:
                x.cost /= base
                x.raw /= base
        results.append(f'{tot} (x{tot / real_tot:.6f})')
        dss.append(alt_items)

    res = [
        ' | '.join(str(dss[j][i]) for j in range(len(dss))) for i in range(len(items))
    ]
    res.append('= ' + ' | '.join(results))
    return '\n'.join(res)
