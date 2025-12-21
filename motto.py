import os
import re
import json
import random
from typing import Iterable

from util import truncate_text, log


def sentences() -> Iterable[str]:
    file = os.environ['MOTTO_FILE']
    with open(file, 'r', encoding='utf-8') as fp:
        s = fp.read()

    for para in s.split():
        splits = re.split(r'([。！？!?])', para)
        for i in range(1, len(splits), 2):
            s = splits[i - 1].strip('「」').strip()
            if s:
                yield s + splits[i]
        if len(splits) % 2 == 1 and (s := splits[-1].strip()):
            if s[-1] == '」' and (s := s.strip('「」').strip()):
                if s[-1] == '，' and len(s) > 1:
                    yield s[:-1] + '。'
                elif s[-1] == '…':
                    yield s
                else:
                    yield s + '。'


SENTENCES = tuple(truncate_text(s) for s in set(s for s in sentences() if len(s) > 5))


def greeting() -> str:
    s = random.choice(SENTENCES)
    log.info('motto: %s', s)
    return s


def cnt():
    from collections import Counter

    return Counter(sentences())


def top(n=None):
    return ''.join(s for s, c in cnt().most_common(n) if c > 2)


SENTENCES_BUNDLE_DIR = os.environ.get('SENTENCES_BUNDLE_DIR', '')


def hitokoto_sentences(sentences_dir: str):
    HITOKOTO_TYPES = os.environ.get('HITOKOTO_TYPES', '').strip()
    HITOKOTO_BANNED = os.environ.get('HITOKOTO_BANNED', '').strip()
    if HITOKOTO_BANNED:
        HITOKOTO_BANNED = tuple(
            s for w in HITOKOTO_BANNED.split(',') if (s := w.strip())
        )
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
                s = (
                    s.replace('······', '……')
                    .replace('......', '……')
                    .replace('...', '…')
                )
                if '。' in s:
                    s = (
                        s.replace(',', '，')
                        .replace(':', '：')
                        .replace(';', '；')
                        .replace('?', '？')
                        .replace('!', '！')
                    )
                yield s


if SENTENCES_BUNDLE_DIR:
    HITOKOTO_SENTENCES = tuple(hitokoto_sentences(SENTENCES_BUNDLE_DIR))
    log.info('Loaded %d hitokoto sentences', len(HITOKOTO_SENTENCES))


def hitokoto() -> str:
    s = random.choice(HITOKOTO_SENTENCES)
    log.info('hitokoto: %s', s)
    return s
