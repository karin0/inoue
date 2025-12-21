import os
import sys
import time
import argparse

if 'TRACE' in os.environ:
    import logging

    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    log.addHandler(logging.FileHandler('render_core_cli.log', 'w', 'utf-8'))

from . import engine
from .engine import Engine

engine.MAX_GAS = 1000000


def try_to_value(s: str) -> int | float | str:
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('-p', '--profile', action='store_true', help='Enable cProfile')
    parser.add_argument(
        '-i',
        '--instrument',
        action='store_true',
        help='Enable pyinstrument',
    )
    parser.add_argument(
        '-c', '--dump-ctx', action='store_true', help='Dump context after rendering'
    )
    parser.add_argument('args', nargs='*')
    args = parser.parse_args()

    file = args.file
    if file == '-':
        text = sys.stdin.read()
    else:
        with open(file, encoding='utf-8') as fp:
            text = fp.read()

    ctx = {str(i): try_to_value(arg) for i, arg in enumerate(args.args)}
    engine = Engine(ctx)

    if args.instrument:
        import pyinstrument

        out_file = 'pyinstrument_report.html'
        fd = os.open(out_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_EXCL)

        with pyinstrument.Profiler() as profiler:
            result, dt = emit(engine, text)

        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(profiler.output_html())
    elif args.profile:
        import cProfile
        import pstats

        with cProfile.Profile() as pr:
            result, dt = emit(engine, text)

        sortby = 'ncalls'
        ps = pstats.Stats(pr, stream=sys.stderr).sort_stats(sortby)
        ps.print_stats()
    else:
        result, dt = emit(engine, text)

    print(result)

    for err in engine.errors:
        print('Error:', err, file=sys.stderr)

    if args.dump_ctx:
        for k, v in engine.items():
            print(f' ', k, '=', repr(v), file=sys.stderr)

    print('Gas used:', engine._gas, file=sys.stderr)
    print(f'Time cost: {dt:.3f} secs', file=sys.stderr)

    sys.exit(bool(engine.errors))


def emit(engine: Engine, text: str):
    t0 = time.perf_counter()
    result = engine.render(text)
    dt = time.perf_counter() - t0
    return result, dt


if __name__ == '__main__':
    main()
