from typing import TYPE_CHECKING

import pytest

from inoue import render_context
from inoue.render_context import OverriddenDict
from render_core import Engine

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from render_core import Value


@pytest.fixture(autouse=True)
def _persisted(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(render_context, 'persisted', {})


def render(text: str, data: Mapping[str, Value] | None = None) -> str:
    engine = Engine(OverriddenDict(data or {}, {}), doc_loader=lambda _: None)
    result = engine.render(text)
    assert not engine.errors, engine.errors
    return result


def primes(limit: int) -> Iterator[int]:
    return (n for n in range(2, limit + 1) if all(n % m for m in range(2, int(n**0.5) + 1)))


def test_override_survives_later_assignment():
    ctx = OverriddenDict({}, {})
    assert Engine(ctx).render('{mode:=write}{mode=read}Current: {mode}') == 'Current: write'
    assert ctx['mode'] == 'write'


def test_override_in_assignment_chains():
    status = {'status': '200'}
    assert render('{c:=d=e=200; c=d=100; status=c=d=200?OK:ERR}', status) == 'OK'
    assert (
        render(
            '{c:=d=e=200; c=d=f=g=100; s=xyxzyqx; s|y/c=e=200|x/c=$f|z/g=$f|q/g=f; '
            'd=c=d=200? $s :ERR; }',
            status,
        )
        == '0101100'
    )
    assert (
        render(
            "{c=d=e='200'; f='100'; x=(c=d=f); y=(c=d=e); z=(c?=d=e); w=(c:=d=$e); x+y+z+w }",
            status,
        )
        == '00200200'
    )
    assert render('a=1; a; x=(a:=2); x; a; y=(a?=3); y; a;') == '12222'


def test_override_keeps_key_order_and_frozen_value():
    ctx = OverriddenDict({}, {})

    def dump(a: int, b: str) -> str:
        return f'{a} {b} {" ".join(ctx)} {" ".join(str(v) for v in ctx.values())}'

    engine = Engine(ctx, funcs=lambda name: dump if name == 'dump' else None)
    text = "a=3; d=7; d:=6; c=4; d=1; b=5; \"dump(42, 'Test')\";"
    assert engine.render(text) == '42 Test a d c b 3 6 4 5'
    assert not engine.errors


def test_pm_scope_persists_across_renders():
    text = '{ a=1; s="\'p\'"; a; s; @("s+\'m\'") { a; x?=::a; x=int(x)+1; x; a=3; a }; a; pm.a; }'
    assert render(text) == '1p12313'

    text = '{ a=1; t=m; s="\'p\'+t"; a; s; @($s) { a; x?=$a; x="int(x)+1"; x; a=2; a }; a; pm.a; }'
    assert render(text) == '1pm33212'


def test_pm_keys_accumulate_across_renders():
    assert render(r't=0; @pm { a?="0"; a+=1; a^t; }; t; pm.a=$t;') == '1'
    assert render(r'c=$pm.a; d="1"; @pm {c=::c; a="c+d"; a;} ;') == '2'
    assert render(r'pm.a?="0"; c=$pm.a; d="1"; @pm {c=::c; a="c+d";}; pm.a;') == '3'

    text = r'pm.a?="0"; c=$pm.a; d="1"; @pm {c=::c; a="c+d";}; "pm.a";'
    assert render(text) == '4'

    # An override on a pm key stays local to the render.
    local = r'@pm {t=$a; a:=11451; a; t; }; a?=810; a;'
    assert render(local) == '114514810'
    assert render(local) == '114514810'
    assert render(text) == '5'

    assert render(r't=@pm {a+=1;a}; t;') == '6'
    assert render(r'@pm{}; ++pm.a; pm.a;') == '7'
    assert render(r'@pm {a+=1;a};') == '8'


def test_pm_generator_yields_primes_until_the_stack_overflows():
    text = r'''{prime_pm:; a=@pm {
n ?= m = "2";
{"m * m > n" ? $n;};
{"m * m > n or n % m == 0" ? n="n+1"; m="2" : m="m+1";};
}; a?"a-1+1":*prime_pm;}
'''
    for v in primes(89):
        assert render(text) == str(v)
    engine = Engine(OverriddenDict({}, {}), doc_loader=lambda _: None)
    engine.render(text)
    assert any('stack overflow' in str(e) for e in engine.errors)


def test_pm_generator_skipping_even_numbers():
    text = r'''{p:; a=@pm {
n ?= m = "2";
{"m * m > n" ? $n; n="n > 2 and n+2 or 3"; m="2" :;};
{"m * m <= n and n % m == 0" ? n="n+2"; m="2" :;};
{"m * m <= n and n % m" ? m="m > 2 and m+2 or 3"};
}; a?$a;:*p;}
'''
    for v in primes(283):
        assert render(text) == str(v)


def test_pm_generator_in_nested_blocks():
    text = r'''p:;
@{ x = @pm {
    n ?= 0;
    n ? {
        { "m*m > n" ? $n; n="n+2"; m="3" :};
        { "n%m == 0" ? n="n+2"; m="3" : m="m+2" };
    } : { "2"; n=m="3"; }
}; x ? 'Result: '; $x : *p;}
'''
    for v in primes(100):
        assert render(text).lstrip('@') == f'Result: {v}'


@pytest.mark.parametrize(
    'text',
    [
        r'''
@{ doc1:; x = @pm {
   n ?= 0;
   n ? "m*m>n" ? $n; n="n+2"; m="3" : !;;;;
       "n%m"   ? m="m+2" : n="n+2"; m="3" !;;;
     : "2"; n=m="3" !;
}; x ? 'Result: '; $x : *doc1; x=0; ! }
x ? rest;
''',
        r'''
@{ doc1:; x = @pm {
   n ?= { n=m="3"; "print('Result: 2\nrest') or exit()" };
   "m*m>n" ? $n; n="n+2"; m="3" : !;
   "n%m"   ? m="m+2" : n="n+2"; m="3" !;
}; x ? 'Result: '; $x : *doc1; x=0; ! }
x ? rest;
''',
    ],
)
def test_pm_generator_with_block_terminators(text: str):
    for v in primes(100):
        assert render(text).lstrip('@') == f'Result: {v}\nrest'
