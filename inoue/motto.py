# ruff: noqa: RUF001, N806
import json
import os
import random
import re

from typing import TYPE_CHECKING

from bot import truncate_text

from .log import is_debug, log

if TYPE_CHECKING:
    from collections.abc import Iterable


def sentences() -> Iterable[str]:
    if not (file := os.environ.get('MOTTO_FILE')):
        yield __name__
        return

    with open(file, encoding='utf-8') as fp:
        s = fp.read()

    for para in s.split():
        splits = re.split(r'([。！？!?])', para)
        for i in range(1, len(splits), 2):
            s = splits[i - 1].strip('「」').strip()
            if s:
                yield s + splits[i]
        if (
            len(splits) % 2 == 1
            and (s := splits[-1].strip())
            and s[-1] == '」'
            and (s := s.strip('「」').strip())
        ):
            if s[-1] == '，' and len(s) > 1:
                yield s[:-1] + '。'
            elif s[-1] == '…':
                yield s
            else:
                yield s + '。'


def greeting() -> str:
    s = random.choice(SENTENCES)  # noqa: S311
    log.info('motto: %s', s)
    return s


def cnt():
    from collections import Counter

    return Counter(sentences())


def top(n=None):
    return ''.join(s for s, c in cnt().most_common(n) if c > 2)


def hitokoto_sentences():
    sentences_dir = os.environ.get('SENTENCES_BUNDLE_DIR')
    if not sentences_dir:
        log.info('SENTENCES_BUNDLE_DIR unset, hitokoto sentences disabled')
        yield __name__
        return

    HITOKOTO_TYPES = os.environ.get('HITOKOTO_TYPES', '').strip()
    HITOKOTO_BANNED = os.environ.get('HITOKOTO_BANNED', '').strip()
    if HITOKOTO_BANNED:
        HITOKOTO_BANNED = tuple(s for w in HITOKOTO_BANNED.split(',') if (s := w.strip()))
    else:
        HITOKOTO_BANNED = ()

    for kind in HITOKOTO_TYPES:
        fn = os.path.join(sentences_dir, kind + '.json')
        with open(fn, encoding='utf-8') as fp:
            for d in json.load(fp):
                assert d['type'] == kind
                s = d['hitokoto'].strip()
                t = d['from'].strip()
                c = [s, t]
                if d_from_who := d['from_who']:
                    t = d_from_who.strip()
                    c.append(t)
                c = ''.join(c)
                if any(w in c for w in HITOKOTO_BANNED):
                    continue
                if t:
                    s += '—— ' + t
                s = s.replace('······', '……').replace('......', '……').replace('...', '…')
                if '。' in s:
                    s = (
                        s.replace(',', '，')
                        .replace(':', '：')
                        .replace(';', '；')
                        .replace('?', '？')
                        .replace('!', '！')
                    )
                yield s


if is_debug:
    HITOKOTO_SENTENCES = SENTENCES = (__name__,)
else:
    HITOKOTO_SENTENCES = tuple(hitokoto_sentences())

    SENTENCES = tuple(truncate_text(s) for s in {s for s in sentences() if len(s) > 5})
    log.info('Loaded %d sentences, %d hitokotos', len(SENTENCES), len(HITOKOTO_SENTENCES))


def hitokoto() -> str:
    s = random.choice(HITOKOTO_SENTENCES)  # noqa: S311
    log.info('hitokoto: %s', s)
    return s
