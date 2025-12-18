import os
import sys
import time
import math
import warnings
import unittest
from unittest.mock import MagicMock

sys.modules['db'] = MagicMock()

db_instance = MagicMock()
sys.modules['db'].db = db_instance

sys.modules['util'] = MagicMock()
sys.modules['util'].shorten = lambda x: x

log_instance = MagicMock()
sys.modules['util'].log = log_instance

if os.environ.get('TEST_TRACE') == '1':
    out = open('test.log', 'w', encoding='utf-8')
    log_effect = lambda m, *a: print(m % a if a else m, file=out)
    log_instance.debug.side_effect = log_effect
    log_instance.info.side_effect = log_effect
    log_instance.warning.side_effect = log_effect
    log_instance.error.side_effect = log_effect
else:
    log_effect = lambda *_: None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from render_core import Value, Engine
from render_core.context import persisted

test_text = """
Hello World
{ user_name = user_nickname = 'Alice' }
{ user_name_o := user_nickname_o = 'Alice' }
{ user_name_2 = 'Alice' }
{ default ?= 'N/A' }
{ +is_login -is_active *qwq }
{ is_login ?
    "'User: ' + user_name"
    :
    is_active ?
    { 'Active user: '; $user_name }
}
{ count := 1; +is_active }
{ count_2 := x; +is_active }
{ user_name }
{ $user_nickname }
{ user_name_o }
{ "default + user_nickname_o" }
{ raw_text = `This is naked`included!the!b`ac`kti`cks! ; $raw_text }
{ a = b = c = d }
{ a ? b } { c ?: d }
this=that; this; `this; {'is'}; `naked;
hello
  the
    wonderful
     美丽新   world.
$this;
a=is also; a; 'naked';
d?=D; e?=E; f?=F;
{ a; b; c; d; e; f; count; count_2; raw_text }
{ doc3:}
{ :doc1 }
{ :doc2 }
{ price="100"; tax="0.08";}
{"float(price) * (1 + float(tax)) > 105"?Expensive: Cheap}
"""


class TestRender(unittest.TestCase):
    def setUp(self):
        db_instance.reset_mock()
        persisted.clear()

    def render_it(
        self,
        text: str,
        ctx: dict[str, Value] | None = None,
        *,
        e: str | tuple[str, ...] = '',
        eq: str | None = None,
    ) -> str:
        if ctx is None:
            ctx = {}
        ctx: Engine = Engine(ctx)
        r = ctx.render(text)
        dump = f'Render: {text}\nResult:{r}\nerrors: {ctx.errors}\nctx: {ctx.items()}'
        log_effect('Gas used: %s', ctx._gas)

        if e:
            if isinstance(e, str):
                e = (e,)
            for e in e:
                self.assertTrue(
                    any(e in str(err) for err in ctx.errors),
                    f'{dump}\nexpected error: {e}',
                )
        else:
            self.assertFalse(
                ctx.errors,
                f'{dump}\nexpected no errors',
            )
        if eq is not None:
            self.assertEqual(r, eq, dump)
        return r

    def render_it_all(
        self, text: str, overrides: dict[str, 'Value'] | None = None
    ) -> tuple[str, Engine]:
        if overrides is None:
            overrides = {}
        ctx = Engine(overrides)
        r = ctx.render(text)
        self.assertFalse(ctx.errors, f'Render: {text} errors: {ctx.errors}')
        return r, ctx

    def test_variable_substitution(self):
        ctx = {'name': 'Alice', 'role': 'Robot'}
        text = "I am {name}, a {role}."
        result = self.render_it(text, ctx)
        self.assertEqual(result, f'I am Alice, a Robot.')

    def test_flags(self):
        result, ctx_on = self.render_it_all(
            "{is_visible?Show:Hide}", {'is_visible': '1'}
        )
        self.assertEqual(result, "Show")

        result, ctx_off = self.render_it_all(
            "{is_visible?Show:Hide}", {'is_visible': '0'}
        )
        self.assertEqual(result, "Hide")
        self.assertFalse(ctx_off.get_flag('is_visible'))
        self.assertTrue(ctx_on.get_flag('is_visible'))

    def test_unary(self):
        self.assertEqual(self.render_it('+a-b+c$a; b; c;'), '101')
        self.assertEqual(self.render_it('k=a; +a-b+c$(k); b; c;'), '101')

        def side_effect(name):
            if name == 'doc':
                return (1, 'd?=2; a; b; c; d;')
            return None

        db_instance.get_doc.side_effect = side_effect

        self.assertEqual(self.render_it('-a-b+c*doc;'), '0012')
        self.assertEqual(self.render_it('+a+b+c; *doc;'), '1112')
        self.assertEqual(self.render_it('+a+b+c;\n:doc;'), '1112')
        self.assertEqual(self.render_it('d=10; +a+b+c*doc*doc*doc;'), '11110' * 3)
        self.assertEqual(
            self.render_it('d=10; +a+b+c;\n:doc; :doc; :doc; *doc;'), '11110' * 4
        )

        self.render_it(
            'a=1; a; @{a; a=7; $a; a=::a; a; c=@q {a; a=3; a; $a; ::a; }; c; a; }; a;',
            eq='1171133111',
        )
        text = '''
.a = "3";
.b = "2";
* @ { a, b ↦
    "a+b"
};
* @ { named ↦
    "a*b"
}; .a; .b;
named ? ERR;
* @ { ↦
    "a*b-a";
    a;
    b;
};
'''
        self.render_it(text, eq='5\n632\n332')
        self.render_it('+{};', e='non-ref')
        self.render_it('*{};', e='non-sub-doc')

    def test_compare(self):
        ctx = {'status': '200'}

        self.assertEqual(self.render_it("{status=200?OK:ERR}", ctx), "OK")
        self.assertEqual(self.render_it("{status==200?OK:ERR}", ctx), "OK")
        self.assertEqual(self.render_it("{status=404?OK:ERR}", ctx), "ERR")
        self.assertEqual(self.render_it("{status==404?OK:ERR}", ctx), "ERR")
        self.assertEqual(self.render_it("{status={ b=200; b } ? OK : ERR}", ctx), "OK")
        self.assertEqual(self.render_it("{status=={ b=200; b } ? OK : ERR}", ctx), "OK")

    def test_assign_or_equal(self):
        ctx = {'status': '200'}
        self.assertEqual(self.render_it("{s=OK}\ns;", ctx), "OK")
        self.assertEqual(
            self.render_it("{s={{ b=200; b } ? OK : ERR}}a\nb;s;", ctx), "a\n200OK"
        )
        self.assertEqual(
            self.render_it("status={; c=\"100+100\"; c} ? OK : ERR;", ctx), "OK"
        )
        self.assertEqual(
            self.render_it("status={; c=\"100+100\"; `200} ? OK : ERR;", ctx), "OK"
        )

        self.assertEqual(
            self.render_it("{c?=d=e=100; c=d=200; status=c=d=200?OK:ERR}", ctx), "OK"
        )
        self.assertEqual(
            self.render_it("{c:=d=e=200; c=d=100; status=c=d=200?OK:ERR}", ctx), "OK"
        )
        self.assertEqual(
            self.render_it(
                "{c:=d=e=200; c=d=f=g=100; s=xyxzyqx; s|y/c=e=200|x/c=$f|z/g=$f|q/g=f; d=c=d=200? $s :ERR; }",
                ctx,
            ),
            "0101100",
        )
        self.assertEqual(
            self.render_it(
                '{c=d=e=200; f=100; x=(c=d=f); y=(c=d=e); z=(c?=d=e); w=(c:=d=$e); "x+y+z+w" }',
                ctx,
                e='bad assign op',
            ),
            "0001",
        )

        # Lazy evaluation is only for `?=`.
        self.assertEqual(self.render_it('a=1; b=1; a={b=2; \'3\'}; a; b;'), '32')
        self.assertEqual(self.render_it('a=1; b=1; a?={b=2; \'3\'}; $a; b;'), '11')
        self.assertEqual(self.render_it('b=1; a?={b=2; \'3\'}; a; $b;'), '32')
        self.assertEqual(self.render_it('a=; b=1; a?={b=2; \'3\'}; a; b;'), '32')
        self.assertEqual(self.render_it('b=1; a?={b=2; \'3\'}; a; b;'), '32')
        self.assertEqual(self.render_it('doc:; a=1; b=1; a?=b=*doc; a; b;'), '11')

        # Dynamic variable names.
        self.assertEqual(self.render_it('a=b; (a)=a=1; $a; $b; '), '11')
        self.assertEqual(self.render_it('a=b; (a)=a=1; c=1; d=b; ((d)=a=$c)?A:B;'), 'A')

    def test_eval(self):
        ctx = {'price': '100', 'tax': '0.08'}
        expr = '{"float(price) * (1 + float(tax)) > 105"?Expensive:Cheap}'
        self.assertEqual(self.render_it(expr, ctx), "Expensive")

        self.assertEqual(self.render_it('"True";'), '1')
        self.assertEqual(self.render_it('"False";'), '0')
        self.assertEqual(self.render_it('"None";'), '')
        self.assertEqual(self.render_it('"\'s\'.encode()";'), 's')
        self.assertEqual(self.render_it('"\'s\'.encode";'), '')

        self.assertAlmostEqual(
            float(self.render_it('"__time__";')), time.time(), delta=2
        )

        self.render_it('"100 ** 100 ** 100 ** 100";', e='Sorry')
        self.render_it('"\'qwq\' * int(1e9)";', e='Sorry')
        self.render_it('a="1 << 100000"; a;', e='Sorry')
        self.render_it('a=2; b="(a:=\'1\')"; a;', e='Sorry')

        # Assignments are not allowed.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertEqual(
                self.render_it(
                    'a=2; b="a=\'1\'"; a;',
                ),
                "2",
            )

        self.render_it('s=qwq; a="(x for x in s)";', e='Sorry')
        self.render_it('s=qwq; a="[x for x in s]";', e='Sorry')
        self.render_it('s=qwq; a="{x:x for x in s}";', e='Sorry')

        # Lists are internally allowed.
        self.assertEqual(
            self.render_it('s=qwqwqwq; a="s.split(\'w\')"; a;'), "['q', 'q', 'q', 'q']"
        )
        self.assertEqual(
            self.render_it('s=qwqwqwq; a="\'z\'.join(s.split(\'w\'))"; a;'), 'qzqzqzq'
        )

        for s in ('lambda x: x', '[]', '()', '(1,)', '{}'):
            self.render_it(f'"{s}";', e='Sorry')

        for s in (
            'list',
            'tuple',
            'dict',
            'set',
            'type',
            'range',
            'help',
            'frozenset',
            'bytearray',
            'memoryview',
            'open',
            'os',
            'sys',
            '__import__',
            'vars',
        ):
            self.render_it(f'a="{s}";', e='not defined')
            self.render_it(f'a="{s}()";', e='not defined')
        for p in (
            '__class__',
            '__bases__',
            '__subclasses__',
            '__globals__',
            '__code__',
            '__module__',
            '__name__',
            '__closure__',
            '__self__',
            '__func__',
            '__dict__',
            '__call__',
        ):
            self.render_it(f'a="{p}";', e='not defined')
            self.render_it(f'a="int.{p}";', e='Sorry')
            self.render_it(f'a="(1).{p}";', e='Sorry')
            self.render_it(f'a="(1,).{p}";', e='Sorry')
            self.render_it(f'a="\'\'.{p}";', e='Sorry')
            self.render_it(f'a="None.{p}";', e='Sorry')

    def test_context_assignment(self):
        text = "{target=World}Hello {target}!"
        result, ctx = self.render_it_all(text)
        self.assertEqual(result, "Hello World!")
        self.assertEqual(ctx.get('target'), 'World')

        self.assertEqual(self.render_it("{a=b=c=3;a;b;c}"), "333")
        self.assertEqual(self.render_it("{a=b=c=3;d=e=;a;d;b;e;c}"), "333")

    def test_context_override(self):
        text = "{mode:=write}{mode=read}Current: {mode}"
        result, ctx = self.render_it_all(text)
        self.assertEqual(result, "Current: write")
        self.assertEqual(ctx.get('mode'), 'write')

    def test_doc_recursion(self):
        def side_effect(name):
            if name == 'header':
                return (1, "Title: {title}")
            return None

        db_instance.get_doc.side_effect = side_effect

        result = self.render_it("{:header}\nBody", {'title': 'My Log'})
        self.assertEqual(result, "Title: My Log\nBody")

    def test_self_recursion(self):
        text = r'''
bomb: ; n ?= "10"; n; n = "n - 1";
n ? *bomb : Boom!;
'''.strip()
        ctx = Engine({})
        result = ctx.render(text)
        a = [str(x) for x in range(10, 0, -1)]
        a.append('Boom')
        self.assertEqual(result, '\n'.join(a))

    def test_circular_dependency(self):
        def side_effect(name):
            if name == 'A':
                return (1, "StartA {:B}")
            if name == 'B':
                return (2, "StartB {:A}")
            return None

        db_instance.get_doc.side_effect = side_effect

        text = "{:A}"
        ctx = Engine({})
        result = ctx.render(text)
        self.assertIn("StartB StartA StartB StartA StartB StartA", result)
        self.assertIn("stack overflow", str(ctx.errors))

    def test_swap(self):
        result, ctx = self.render_it_all("{x=1; y=2; x^y; x; y}")
        self.assertEqual(result, "21")
        self.assertEqual(ctx.get('x'), '2')
        self.assertEqual(ctx.get('y'), '1')

        result = self.render_it("{a=hello; a^b; a}", e='undefined')
        self.assertEqual(result, '')

        result = self.render_it("{a=hello; a^b; $b}")
        self.assertEqual(result, 'hello')

    def test_quine(self):
        q = r'''s='s=%r; "s %% s";'; "s % s";'''
        self.assertEqual(self.render_it(q), q)
        q = r'''s=`"'s=`' + s + '; ' + s + ';'"; "'s=`' + s + '; ' + s + ';'";'''
        self.assertEqual(self.render_it(q), q)
        q = r'''
Write the following sentence twice, the second time within quotes.
"Write the following sentence twice, the second time within quotes."
'''.strip()
        self.assertEqual(self.render_it(q), q)
        q = r'#/bin/cat'
        self.assertEqual(self.render_it(q), q)
        q = r'''"__file__";'''
        self.assertEqual(self.render_it(q), q)
        self.assertEqual(self.render_it(q * 2), q * 4)
        self.assertEqual(self.render_it(q * 3), q * 9)

    def test_replace(self):
        self.assertEqual(self.render_it('s=114514; s|4/9; s;'), '119519')
        self.assertEqual(self.render_it('s=hello world; s|world/; s;'), 'hello')
        self.assertEqual(self.render_it('s=hello; s|/1; s;'), '1h1e1l1l1o1')
        self.assertEqual(self.render_it('s=hello; s|h/|l/|o/|e/; s;'), '')
        self.assertEqual(
            self.render_it(
                's=hello; s|h/{a=$s;}|l/{b=$s;}|{({`o})}/{c=$s;}|e/{d=$s}; s;a;b;c;d;'
            ),
            'helloelloeoe',
        )

    def test_empty_values(self):
        text = '{;a=} {;b:=;} {c=; d=2345;} {d|4/} {"a+b+c+d"}'
        result = self.render_it(text)
        self.assertEqual(result, '235')

    def test_nested_blocks(self):
        text = '{ a=1; { b=2; { c=3; "a + b + c" }; "a + b" }; a }'
        result = self.render_it(text)
        self.assertEqual(result, '123121')

    def test_nested_branches(self):
        result = self.render_it(
            '{ {x="10"; "x>5" ? y="20"; "y>15" ? High : Medium : Low} ;` rest }'
        )
        self.assertEqual(result, 'High rest')
        result = self.render_it(
            '{ x="0"; "x>5" ? y="20"; "y>15" ? High : Medium : Low ; `rest }'
        )
        self.assertEqual(result, 'Lowrest')

        result = self.render_it(
            '{ x="100"; "x>5" ? y="20"; "y>15" ? High : Medium : Low ; `rest }'
        )
        self.assertEqual(result, 'Highrest')

        result = self.render_it('{ x="100" ? }\na')
        self.assertEqual(result, 'a')

        result = self.render_it('{ x="100" ?: }\na')
        self.assertEqual(result, 'a')

    def test_nested_blocks_and_branches(self):
        text = '{ a="1"; "a==1"? { b="2"; "a+b==3"? { c="3"; "a+b+c" } : "Wrong" } : "Wrong" }'
        result = self.render_it(text)
        self.assertEqual(result, '6')

    def test_nested_expressions(self):
        text = '{ a = "30"; b = "12"; c = ( ( "a + b" ) ) ; c }'
        result = self.render_it(text)
        self.assertEqual(result, '42')
        self.assertEqual(self.render_it('a=42; { ( ( ( a ) ) ) };'), '42')

    def test_doc_name_definition(self):
        def f(text):
            ctx = Engine({})
            ctx.render(text)
            return ctx.doc_name

        text = '{ doc_test: } This is doc.'
        self.assertEqual(f(text), 'doc_test')

        text = '{ doc_test:; } This is doc.'
        self.assertEqual(f(text), 'doc_test')

        text = '{ doc_test:; qwqw } {:doc_test} '
        self.assertEqual(f(text), 'doc_test')

        text = 'hello!\ndoc_test:; qwqw;\n{:doc_test}'
        self.assertEqual(f(text), 'doc_test')

    def test_scope(self):
        self.assertEqual(self.render_it('{ a=1; @s { a=2; a }; a; s.a; }'), '212')
        self.assertEqual(
            self.render_it(
                '{ a=1; a; @s { a; b=::a; a=2; a; @s { a; a=3; a }; s.a; a; b; }; s.s.a; s.a; s.b; a; @{ a; a=7; a; }; a; .a; }'
            ),
            '1122332132111717',
        )

        # Mutating outer scope using replacement.
        self.assertEqual(
            self.render_it(
                '{ a=124524; a; @s { a; a|2/1; a; b=::a; }; a; s.b; "s.b" }'
            ),
            '124524' * 2 + '114514' * 4,
        )

        # Mutating outer scope using swapping.
        self.assertEqual(
            self.render_it('{ a=810; a; @s { a; b=893; b; a^b; a; b; }; s.b; a; }'),
            '810' * 2 + '893' * 2 + '810810893',
        )

        # Paren as scope name.
        text = '{ a=1; s="\'p\'"; a; s; @("s+\'m\'") { a; x?=::a; x="int(x)+1"; x; a=3; a }; a; pm.a; }'
        self.assertEqual(self.render_it(text), '1p12313')

        text = '{ a=1; t=m; s="\'p\'+t"; a; s; @($s) { a; x?=$a; x="int(x)+1"; x; a=2; a }; a; pm.a; }'
        self.assertEqual(self.render_it(text), '1pm33212')

        # Scope with empty name is not global.
        text = '{ a=1; a; @ {a; b=2; a; b; a=3; a; b; a^b; a; b; b=::a; b; @{c={b}} }; a; .a; .b; ..c; pm.a; }'
        self.assertEqual(self.render_it(text), '11123223112112')

        # Scope declaration inside block.
        text = '{ @s; a=1; a; {@r; b=1; b; }; b?=3; b; r.b; }'
        self.assertEqual(self.render_it(text), '1131')

        # Scope declaration inside block.
        text = '{ @s; a=1; a; x=r; {@(x); b=1; b; }; b?=3; b; r.b; }'
        self.assertEqual(self.render_it(text), '1131')

        text = '{ x=1; @s {@(x); b=1; b; }; }'
        self.render_it(text, e='double scope')

    def test_static(self):
        text = r't=0; @pm { a?="0"; a="a+1"; a^t; }; t; pm.a=$t;'
        self.assertEqual(self.render_it(text), '1')

        text = r'c=$pm.a; d="1"; @pm {c=::c; a="c+d"; a;} ;'
        self.assertEqual(self.render_it(text), '2')

        text = r'pm.a?="0"; c=$pm.a; d="1"; @pm {c=::c; a="c+d";}; pm.a;'
        self.assertEqual(self.render_it(text), '3')

        text = r'pm.a?="0"; c=$pm.a; d="1"; @pm {c=::c; a="c+d";}; "pm.a";'
        self.assertEqual(self.render_it(text), '4')

        # PM keys can be overridden, but that makes them local and doesn't affect
        # persisted ones.
        text2 = r'@pm {t=$a; a:=11451; a; t; }; a?=810; a;'
        self.assertEqual(self.render_it(text2), '114514810')
        self.assertEqual(self.render_it(text2), '114514810')
        self.assertEqual(self.render_it(text), '5')

        text = r't=@pm {a="a+1";a}; t;'
        self.assertEqual(self.render_it(text), '6')

        text = r'@pm{}; pm.a="pm.a+1"; pm.a;'
        self.assertEqual(self.render_it(text), '7')

        text = r'@pm {a="a+1";a};'
        self.assertEqual(self.render_it(text), '8')

    def test_return_value(self):
        # The type of the return value must be preserved if only one value is
        # yielded from a `code_block` or `unary_chain`.
        self.assertEqual(self.render_it('{ a={b="1"; b}; "a+1" }'), '2')
        self.assertEqual(self.render_it('{ a={d={{"1"}}; +b-c$d}; "a+1" }'), '2')
        self.assertEqual(
            self.render_it(r'''{ a={d={{"1"}}; {d=="1"?q}; +b-c$d}; "a+'1'" }'''),
            'q11',
        )

    def test_raw_text(self):
        text = '{ a =` This is naked`?=i:n$cluded!the!b`ac`kti`c\nks! ; "1"; a; "2" }'
        result = self.render_it(text)
        self.assertEqual(result, '1 This is naked`?=i:n$cluded!the!b`ac`kti`c\nks! 2')

    def test_common(self):
        def side_effect(name):
            if name == 'qwq':
                return (1, "Default: {default} {$default}")
            elif name == 'doc1':
                return (2, "This is doc1:\na;b;\nc;\n")
            elif name == 'doc2':
                return (3, "This is doc2:\nthis;\n")
            return None

        db_instance.get_doc.side_effect = side_effect

        result, ctx = self.render_it_all(test_text)
        answer = 'Hello World\n\n\n\n\nDefault: N/A N/A\nUser: Alice\n\n\nAlice\nAlice\nAlice\nN/AAlice\nThis is naked`included!the!b`ac`kti`cks! \n\nb \nthatthisisnaked\nhello\n  the\n    wonderful\n     美丽新   world.\nthat\nis alsonaked\nis alsoddDEF1xThis is naked`included!the!b`ac`kti`cks! \n\nThis is doc1:\nis alsod\nd\n\nThis is doc2:\nthat\n\n\nExpensive'
        self.assertEqual(result, answer)
        self.assertEqual(ctx.doc_name, 'doc3')

    def test_plain(self):
        text = "This is a plain text without any variables."
        result = self.render_it(text)
        self.assertEqual(result, text)

    def test_escape(self):
        text = r"""{ text = 'He said: \'Hello, World!\'New line.'; text }"""
        result = self.render_it(text)
        self.assertEqual(result, "He said: 'Hello, World!'New line.")

        text = r"""{ text = "'She replied: \"Hi there!\"Another line.'"; text }"""
        result = self.render_it(text)
        self.assertEqual(result, 'She replied: "Hi there!"Another line.')

        text = r"""{ text = `Raw string with 'single;' and "}double{" quotes.; text }"""
        result = self.render_it(text)
        self.assertEqual(
            result, r'''Raw string with 'single;' and "}double{" quotes.'''
        )

        text = r"""{ text = 'Mixing \'escaped;\' and `raw` quotes.'; text }"""
        result = self.render_it(text)
        self.assertEqual(result, "Mixing 'escaped;' and `raw` quotes.")

        text = r"""{ text = `Mixing 'raw;' and \"unescaped\" quotes.; text }"""
        result = self.render_it(text)
        self.assertEqual(result, r'''Mixing 'raw;' and \"unescaped\" quotes.''')

        text = r"""{ text = 'Escaped backslash: \\ and quote: \''; text }"""
        result = self.render_it(text)
        self.assertEqual(result, "Escaped backslash: \\ and quote: '")

        text = r'\{ not a block ! \}'
        self.render_it(text, eq='{ not a block ! }')
        self.render_it(text + '\nabc', eq='{ not a block ! }\nabc')

        text = r'not a block either ! \;'
        self.render_it(text, eq='not a block either ! ;')
        self.render_it(text + '\nabc', eq='not a block either ! ;\nabc')

        text = r'backslash as text \\  \;'
        self.render_it(text, eq=r'backslash as text \  ;')

    def test_unclosed(self):
        text = r"""{ text = `Raw backslash: \ and quote: '`; text }"""
        result = self.render_it(text, e='Unclosed')
        self.assertEqual(result, text.replace('\\', ''))

        text = r''' { another = unclosed = block = "123" '''
        result = self.render_it(text, e='Unclosed')
        self.assertEqual(result, text.strip())

        text = r'''a=1; a;'''
        result = self.render_it(text)
        self.assertEqual(result, '1')

        # Naked block across multiple lines.
        text = r'''a=1; a = {
c={
    '51';
    {
        '4';
    }
};
"'114' + c";
};
{
a
};
@ns {
    b=$a;
    b;
};
ns.b;
'''
        result = self.render_it(text)
        self.assertEqual(result + '\n', '114514\n' * 3)

        text = r''' {1 {'2'} {3{4 {"'partly'"}; 'closed' '''
        result = self.render_it(text, e='Unclosed')
        self.assertEqual(result, "{1 2 {3{4 partly; 'closed'")

        text = r'''Not block \{ 'is here!' \}'''
        result = self.render_it(text)
        self.assertEqual(result, text.replace('\\', ''))

        text = r"'qwqwq'; } '123';"
        result = self.render_it(text, e='Unexpected token')
        self.assertEqual(result, '')

        text = r"'1'; { '2'; { '3'; } {{{{{ '4'; };"
        result = self.render_it(text, e=('Unbalanced', 'Unexpected token'))
        self.assertEqual(result, '')

    def test_prime(self):
        text = r'''prime:;
n ?= m = "2";
"m * m > n" ? $n;
"m * m > n or n % m == 0" ? n="n+1"; m="2" : m="m+1";
:prime;
'''
        result = self.render_it(text, e='stack overflow')
        self.assertTrue(result.startswith('2\n3\n5\n7\n11\n13'), result)

    def test_prime_2(self):
        text = r'''prime:;
{n ? "m*m>n" ? $n; n="n+2"; m="3"        !;
      "n%m"   ? m="m+2" : n="n+2"; m="3"; !
    : 2; n=m="3"; !; }
*prime;
'''
        result = self.render_it(text, e='stack overflow')
        result = '\n'.join(result.split())
        self.assertTrue(result.startswith('2\n3\n5\n7\n11\n13'), result)

    def test_prime_3(self):
        text = r'''prime2:;
n ?= m = "2";
"m * m > n" ? $n; n="n > 2 and n+2 or 3"; m="2" :;
"m * m <= n and n % m == 0" ? n="n+2"; m="2";   :;
"m * m <= n and n % m" ? m="m > 2 and m+2 or 3" :;
*prime2;
'''
        result = self.render_it(text, e='stack overflow')
        self.assertTrue(result.startswith('2\n3\n5\n7\n11\n13'), result)

    def test_prime_vectorized(self):
        db = {}
        db[
            'iter0'
        ] = r'''
iter: ;
"m * m <= n and n % m == 0" ? n="n+2"; m="2"; :;
"m * m <= n and n % m" ? m="m > 2 and m+2 or 3";
"m * m > n" ? $n; n="n+2";  m="2" :;
'''
        db['iter1'] = 'iter1:;' + '*iter0;' * 100
        db['iter2'] = 'iter2:;' + '*iter1;' * 100
        db['iter3'] = 'iter3:;' + '*iter2;' * 100
        db['iter4'] = 'iter4:;' + '*iter3;' * 100

        def side_effect(name):
            if doc := db.get(name):
                return (1, doc)
            return None

        db_instance.get_doc.side_effect = side_effect

        text = (
            r'''
        prime2: ; n ?= m = "2";
"m * m > n" ? $n; n="n > 2 and n+2 or 3"; m="2" :;
'''
            + '*iter4;' * 100
            + '*prime2;'
        )
        result = self.render_it(text, e='out of gas')
        ans = '\n'.join(str(v) for v in self.iter_prime(59))
        self.assertTrue(result.startswith(ans), result)

    @staticmethod
    def iter_prime(limit: int):
        yield 2
        n = 3
        while n <= limit:
            for m in range(3, int(n**0.5) + 1):
                if n % m == 0:
                    break
            else:
                yield n
            n += 2

    def test_prime_pm(self):
        # Use PM to persist state across multiple invocations, like a "generator".
        text = r'''{prime_pm:; a=@pm {
n ?= m = "2";
{"m * m > n" ? $n;};
{"m * m > n or n % m == 0" ? n="n+1"; m="2" : m="m+1";};
}; a?"a-1+1":*prime_pm;}
'''
        for v in self.iter_prime(89):
            self.assertEqual(self.render_it(text), str(v))
        self.render_it(text, e='stack overflow')

    def test_prime_pm2(self):
        text = r'''{p:; a=@pm {
n ?= m = "2";
{"m * m > n" ? $n; n="n > 2 and n+2 or 3"; m="2" :;};
{"m * m <= n and n % m == 0" ? n="n+2"; m="2" :;};
{"m * m <= n and n % m" ? m="m > 2 and m+2 or 3"};
}; a?$a;:*p;}
'''
        for v in self.iter_prime(283):
            self.assertEqual(self.render_it(text), str(v))

    def test_prime_pm3(self):
        text = r'''p:;
@{ x = @pm {
    n ?= 0;
    n ? {
        { "m*m > n" ? $n; n="n+2"; m="3" :};
        { "n%m == 0" ? n="n+2"; m="3" : m="m+2" };
    } : { "2"; n=m="3"; }
}; x ? 'Result: '; $x : *p;}
'''
        for v in self.iter_prime(100):
            self.assertEqual(self.render_it(text).lstrip('@'), 'Result: ' + str(v))

    def test_prime_pm4(self):
        text1 = r'''
@{ doc1:; x = @pm {
   n ?= 0;
   n ? "m*m>n" ? $n; n="n+2"; m="3" : !;;;;
       "n%m"   ? m="m+2" : n="n+2"; m="3" !;;;
     : "2"; n=m="3" !;
}; x ? 'Result: '; $x : *doc1; x=0; ! }
x ? rest;
'''
        text2 = r'''
@{ doc1:; x = @pm {
   n ?= 0;
   n ? "m*m>n" ? $n; n="n+2"; m="3" : !;;;;
       "n%m"   ? m="m+2" : n="n+2"; m="3" !;;;
     : "2"; n=m="3" !;
}; x ? 'Result: '; $x : *doc1; x=0; ! }
x ? rest;
'''

        text3 = r'''
@{ doc1:; x = @pm {
   n ?= { n=m="3"; "print('Result: 2\nrest') or exit()" };
   "m*m>n" ? $n; n="n+2"; m="3" : !;
   "n%m"   ? m="m+2" : n="n+2"; m="3" !;
}; x ? 'Result: '; $x : *doc1; x=0; ! }
x ? rest;
'''

        for text in (text1, text2, text3):
            persisted.clear()
            for v in self.iter_prime(100):
                self.assertEqual(
                    self.render_it(text).lstrip('@'), 'Result: ' + str(v) + '\nrest'
                )

    def test_fib_wrong(self):
        # Use scopes as "stack frames" to achieve non-tail recursion!
        # By wrapping the whole block in a scope, we can protect any "local" variables
        # effectively from being clobbered by our "callees" (as long as they also
        # use scopes and avoid swaps/repls), making it *look like* a real "function".
        # However, that might be the only guarantee we can get, and there are
        # still the following caveats:
        # 1. `ScopedContext` also searches names from outer scopes;
        # 2. `ScopedContext` does NOT clear the scope when pushing/popping our "stack frames".
        # As such, when we are the "callee", we have to be careful to "untrust"
        # the state of the current scope ("uninitialized local variables!"),
        # since the "stack frame" could be dirty if the stack have "grown" again
        # to reuse an existing scope.
        # Due to these reasons, when we read a variable in the inner scope, there
        # is no way to be sure if it is a "local" variable, or from outer scopes.
        # This makes a direct '?=' misbehave without rebinding the outer "arguments"
        # to new local variables first.
        # Welcome back to the C world, but with LEGB rules this time!
        text = r'''{@; fib:;
# We meant to read the outer `n`, but we cannot get it reliably.
# n ?= $n;     # Wrong! "$n" may (and may not) refer to the current (dirty) scope!
# n='' ? n=$n; # Still wrong!
n ?= "7";      # Default argument. But, the legacy value in the *current* scope
               # could block anything, either the default value or the outer `n`.
"n<=1" ? $n :  # Flawed: a legacy value from previous call lurks here!
  n = "n-1" ;
  a = *fib  ;
  n = "n-1" ;  # Fortunately, at least scope shadowing allows us to reuse n safely.
  b = *fib  ;  # Trouble is for the "callee" here, which uses the same scope as
               # previous call, making the "stack frame" dirty inside.
  "a + b"   !
}'''
        # A wrong implementation that doesn't rebind `n` yields wrong results.
        self.assertEqual(self.render_it(text), '7')

    def test_fib(self):
        text = r'''{fib:;
n ?= "7";
.n = $n;        # Before entering our scope, set the argument there explicitly!
@{
  "n<=1" ? $n :
    n = "n-1" ;
    a = *fib  ;
    n = "n-1" ;
    b = *fib  ;
    "a + b"   !
}}'''
        self.assertEqual(self.render_it(text), '13')
        self.render_it(text.replace('7', '10'), e='out of gas')

    def test_fib_single_scope(self):
        text = r'''{@; fib:;
n ?= "7";
"n<=1" ? $n  :
  .n = "n-1" ; # Write to the callee's scope explicitly to work around the issue,
               # like we are passing a "keyword argument" by value explicitly.
  a = *fib   ;
  .n = "n-2" ; # Not "n-1" this time!
  b = *fib   ;
  "a + b"    !
}'''.strip()
        self.assertEqual(self.render_it(text), '13')
        self.render_it(text.replace('7', '10'), e='out of gas')

    def test_fib_rebind(self):
        # Magic: '::' operator to refer to the outer scope explicitly.
        # Another "calling convention", looks cleaner but less explicit.
        text = r'''{@; fib:
n = ::n;    # Rebind the outer "argument" to initialize a "local" variable!
n ?= "9";
"n<=1" ? $n :
  n = "n-1" ;
  a = *fib  ;
  n = "n-1" ;
  b = *fib  ;
  "a + b"   !
}'''
        self.assertEqual(self.render_it(text), '34')
        self.render_it(text.replace('9', '10'), e='out of gas')

    def test_sub_doc(self):
        # This simpler "calling convention" doesn't need to read from outer scopes.
        text = r'''{@; fib ↦
n = $a0;
"n<=1" ? $n :
  .a0 = "n-1" ;
  a = *fib  ;
  .a0 = "n-2" ;
  b = *fib  ;
  "a + b"   !
}
.a0="9"; *fib;
'''
        self.assertEqual(self.render_it(text), '34')
        self.render_it(text.replace('9', '10'), e='out of gas')

    def test_sub_doc_2(self):
        text = r'''{
@{fib ↦
  n = ::n;
  "n<=1" ? $n :
    n = "n-1" ;
    a = *fib  ;
    n = "n-1" ;
    b = *fib  ;
    "a + b"   !
};
n="9"; *fib;
}
'''
        self.assertEqual(self.render_it(text), '34')
        self.render_it(text.replace('9', '10'), e='out of gas')

    def test_sub_doc_3(self):
        text = r'''{@; fib ↦
n = ::n;
"n<=1" ? $n :
  n = "n-1" ;
  a = *fib  ;
  n = "n-1" ;
  b = *fib  ;
  "a + b"   !
}
n="9"; *fib;
'''
        self.assertEqual(self.render_it(text.replace('9', '7')), '13')
        self.assertEqual(self.render_it(text), '34')
        self.render_it(text.replace('9', '10'), e='out of gas')

    def test_lambda(self):
        text = r'''{foo = {@; ↦
n = ::n;
"n<=1" ? $n :
  n = "n-1" ;
  a = *foo  ;
  n = "n-1" ;
  b = *foo  ;
  "a + b"   !
}}
n="9"; *foo;
'''
        self.assertEqual(self.render_it(text), '34')

    def test_lambda_2(self):
        text = r'''
fib = {@; x ↦ "x > 1 and (fib(x-1) + fib(x-2)) or x" };
"fib(9)";
.x="7"; *fib;
'''
        self.assertEqual(self.render_it(text), '34\n13')

    def test_lambda_3(self):
        text = r'''
n="2"; m="1"; a=0;
a ? f={ ↦ $n } : f={ ↦ "m" };
*f;
a="42";
a ? f={ ↦ "n" ;} : f={ ↦ $m; };
*f;
f = {@; a, b ↦ "print(a, n, m) or a * b * t" };
"f(n+m, 7, t=2)";
f = @ { x ↦ "x * (x+1)" };
x = "6"; *f;
a;
g = {↦};
"g() is None";
{@; h ↦ "n > 1 and g(n=n-1) * n or 1" };
g^h; b="g(n=5)"; b;
f = @ { ↦ a=$0; b=$1; "a + b" };
"f(a, b)";
'''
        self.assertEqual(self.render_it(text), '1\n2\n3 2 1\n42\n42\n42\n1\n\n120\n162')

    STD = r'''
wile = while = {cond, body ↦
  *cond ? *body *while
};
fur = for = {init, cond, step, body ↦
  *init ;
  _body = $body;
  body = { ↦ *_body *step };
  *while
};
'''

    def test_loop(self):
        text = r'''
n = "1"; out=;
{ body ↦ out = "out + str(n) + ' '"; n = "n+1" };
{ cond ↦ "n<10" };
*while; a=$out; out=; n = "1";
"wile(cond, body)"; b=$out;
a; b; a = $b ? Ok : Fail;'''
        self.render_it(self.STD + text, eq='1 2 3 4 5 6 7 8 9 ' * 2 + 'Ok')

    def test_for_loop(self):
        text = r'''
{ init ↦ n = "1"; out=; };
{ cond ↦ "n<=9" };
{ step ↦ n = "n+1" };
{ body ↦ out = "out + str(n) + ' '" };
*for; a = $out; a;'''
        self.render_it(self.STD + text, eq='1 2 3 4 5 6 7 8 9')

    def test_sort(self):
        ctx = [5, 3, 8, 6, 2, 7, 4, 1]
        ctx = {f'a{i}': str(v) for i, v in enumerate(ctx)}
        text = r'''
n="8";
{ init ↦ i = "0" };
{ cond ↦ "i<n" };
{ step ↦ i = "i+1" };
{ body ↦ ({'v'; i}) = $({'a'; i}) };
*for;

{@; body ↦
    { init ↦ j = "0" };
    { cond ↦ "j < n-i-1" };
    { step ↦ j = "j+1" };
    { body ↦
        a = $({'v'; j}); b = $({'v'; "j+1"});
        "a>b" ? ({'v'; j}) ^ ({'v'; "j+1"});
    };
    *for;
}
*for;

{ body ↦ x = $({'v'; i}) ; "print(x)" };
"fur(init, cond, step, body)";
'''
        self.render_it(self.STD + text, ctx, eq='\n'.join(str(i) for i in range(1, 9)))

    def test_tco(self):
        text = r'''
# tco test;
r = @ {
    n = "100";
    { init ↦ i = "0" };
    { cond ↦ "i<n" };
    { step ↦ i = "i+1" };
    { body ↦ };
    *for;
};
.i = 100 ? Ok : Fail;
'''
        self.render_it(self.STD + text, eq='Ok')

    def test_tco_fact(self):
        text = r'''
fac = {
@; n, m ↦
    "n>1" ?
      .n = "n-1";
      .m = "(n-1)*m";
      *fac
    : $m ;
};
"fac(10, 10)";
.n = .m = "50";
*fac;
'''

        ans = math.factorial(10)
        ans2 = math.factorial(50)
        self.render_it(text, eq=f'{ans}\n{ans2}')

    def test_tco_ffi(self):
        text = r'''
fac = {
@; n, m ↦
    m ?= "1";
    "n>1" ?
      "fac(n-1, n*m)";
    : $m ;
};
"fac(50)";
'''
        ans = math.factorial(50)
        self.render_it(text, eq=str(ans))

    def test_subscript(self):
        text = r'''
i = "1";
a[i] = "7";
a[i];
a.1;
j = 6;
a[j] = "42";
a[j] ^ a[i];
a[i];
a[j];
k = "7";
a[k] = 11;
"a[i] + a[j] + int(a[k][1])";
'''
        self.render_it(text, eq='7\n7\n42\n7\n50')


if __name__ == '__main__':
    unittest.main()
