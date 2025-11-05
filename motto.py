import os
import re
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
    log.info('greeting: %s', s)
    return s


def cnt():
    from collections import Counter

    return Counter(sentences())


def top(n=None):
    return ''.join(s for s, c in cnt().most_common(n) if c > 2)
