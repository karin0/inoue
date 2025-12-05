from math import floor
from typing import Hashable
from collections import defaultdict

from arithmetic_eval import evaluate
from prettytable import PrettyTable


def calc(s: str) -> float:
    return evaluate(s, {'x': 1.1})


def render_sign(v: float | int) -> str:
    if isinstance(v, int):
        return f'{v:+}'
    return f'{v:+.3}'


def render_pair(x, y) -> tuple[str, str]:
    if y == 0:
        return str(x), ''
    return str(x), f'({render_sign(y)})'


class Value:
    def __init__(self, raw: float, kind: Hashable = None, comment: str = ''):
        self.raw = raw
        self.cost = round(raw)
        self.kind = kind  # Total cost is rounded for each group by kind
        self._comment = comment

    @property
    def adj(self):
        return self.cost - self.raw

    @property
    def comment(self):
        return f'# {self._comment}' if self._comment else ''

    def render(self):
        return render_pair(self.cost, self.adj)


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
        comment = ''
        if (p := line.find('#')) >= 0:
            comment = line[p + 1 :].strip()
            line = line[:p]
        if not (line := line.strip()):
            continue
        cmd = line[0]
        if not (cmd.isdigit() or cmd == '(' or cmd == '.'):
            arg = line[1:].strip()
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

        items.append(Value(calc(line) * tax, tax, comment))

    if not items:
        raise ValueError('No values provided')

    tot_dict = defaultdict(float)
    for item in items:
        tot_dict[item.kind] += item.raw
    real_tot = sum(floor(v) for v in tot_dict.values())

    warn = None
    ext_idx = None
    if the_real_tot is not None and the_real_tot != real_tot:
        if the_real_tot > real_tot:
            ext_idx = len(items)
            delta = the_real_tot - real_tot
            items.append(Value(delta))
            warn = f'⚠️ {the_real_tot} (expected) > {real_tot}, added {delta}'
            real_tot = the_real_tot
        else:
            warn = f'❌ {the_real_tot} (expected) < {real_tot}'

    alter(items, real_tot)
    tots = list(render_pair(real_tot, real_tot - sum(item.raw for item in items)))

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
            tots.append(str(tot))
            tots.append(f'(x{tot / real_tot:.3f})')
            dss.append(alt_items)

    row_len = len(dss) << 1 | 1
    fields = [str(i) for i in range(row_len)]
    aligns = {str(i): ('l' if i & 1 else 'r') for i in range(row_len)}
    aligns[str(row_len - 1)] = 'l'

    table = PrettyTable(
        fields, header=False, border=False, preserve_internal_border=True, align=aligns
    )
    for i, row in enumerate(zip(*dss)):
        row: list[Value]
        comment = row[0].comment
        if i == ext_idx:
            comment += '🚨'

        r = []
        for x in row:
            r.extend(x.render())
        r.append(comment)
        table.add_row(r)

    tots[0] = '= ' + tots[0]
    tots.append('#inoue')
    table.add_row(tots)

    res = str(table)
    if warn:
        res += f'\n{warn}'

    return res
