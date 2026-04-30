import atexit
import logging
from typing import Iterable, Sequence, cast
from sqlite3 import connect, Connection

from context import ME_LOWER

log = logging.getLogger(ME_LOWER + '.db')


class DataStore:
    __slots__ = ('conn',)

    def __init__(self):
        self.conn = cast(Connection, None)

    def connect(self, file: str):
        if self.conn is not None:
            raise RuntimeError('Already connected')

        self.conn = connect(file, autocommit=False)
        atexit.register(self.conn.close)

        with self.conn:
            self.conn.executescript('''CREATE TABLE IF NOT EXISTS Doc (
                    id    INTEGER PRIMARY KEY NOT NULL,
                    name  TEXT    UNIQUE NOT NULL,
                    text  TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS KV (
                    id    INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                    key   TEXT    UNIQUE NOT NULL,
                    value TEXT    NOT NULL
                );

                CREATE TRIGGER IF NOT EXISTS KV_fifo
                AFTER INSERT ON KV
                BEGIN
                    DELETE FROM KV WHERE id IN (
                        SELECT id FROM KV ORDER BY id DESC LIMIT -1 OFFSET 1000
                    );
                END;

                CREATE TABLE IF NOT EXISTS Command (
                    id    INTEGER PRIMARY KEY NOT NULL,
                    name  TEXT    UNIQUE NOT NULL,
                    cmd   TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS Media (
                    id         INTEGER PRIMARY KEY NOT NULL,
                    chat_id    INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    title      TEXT    NOT NULL,
                    file_id    TEXT    NOT NULL,
                    saved_at   TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(chat_id, message_id)
                );
                ''')
        log.debug('Connected to db: %s', file)

    def close(self):
        c = self.conn
        self.conn = cast(Connection, None)
        (f := c.close)()
        atexit.unregister(f)

    def summary(self) -> str:
        n = self.conn.execute('SELECT COUNT(*) FROM Doc').fetchone()[0]
        m = self.conn.execute('SELECT COUNT(*) FROM KV').fetchone()[0]
        c = self.conn.execute('SELECT COUNT(*) FROM Command').fetchone()[0]
        s = self.conn.execute('SELECT COUNT(*) FROM Media').fetchone()[0]
        return f'{n} docs, {m} keys, {c} commands, {s} media'

    def get_doc(self, name: str) -> tuple[int, str] | None:
        cursor = self.conn.execute('SELECT id, text FROM Doc WHERE name = ?', (name,))
        return cursor.fetchone()

    # Works like INSERT OR REPLACE, but returns the replaced records.
    def save_doc(
        self, id: int, name: str, text: str
    ) -> tuple[tuple[str, str] | None, tuple[int, str] | None]:
        with self.conn:
            row_by_id = self.conn.execute(
                'SELECT name, text FROM Doc WHERE id = ?;', (id,)
            ).fetchone()
            if row_by_id and row_by_id[0] == name:
                row_by_name = None
            elif row_by_name := self.get_doc(name):
                self.conn.execute('DELETE FROM Doc WHERE name = ?;', (name,))

            self.conn.execute(
                '''INSERT INTO Doc (id, name, text) VALUES (?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET name = excluded.name, text = excluded.text;''',
                (id, name, text),
            )

        return row_by_id, row_by_name

    def delete_doc(self, id: int) -> tuple[str, str] | None:
        with self.conn:
            if row := self.conn.execute(
                'SELECT name, text FROM Doc WHERE id = ?;', (id,)
            ).fetchone():
                self.conn.execute('DELETE FROM Doc WHERE id = ?;', (id,))

        return row

    def find_docs(self, kws: Sequence[str]) -> Iterable[tuple[int, str, int]]:
        if kws:
            cond = ' OR '.join('name LIKE ?' for _ in kws)
            query = f'SELECT id, name, LENGTH(text) FROM Doc WHERE {cond} ORDER BY id;'
            args = tuple(f'%{kw}%' for kw in kws)
            yield from self.conn.execute(query, args)
        else:
            query = 'SELECT id, name, LENGTH(text) FROM Doc ORDER BY id;'
            yield from self.conn.execute(query)

    def get(self, key: str, default: str | None = None) -> str | None:
        cursor = self.conn.execute('SELECT value FROM KV WHERE key = ?;', (key,))
        if row := cursor.fetchone():
            return row[0]
        return default

    def __getitem__(self, key: str) -> str:
        if value := self.get(key):
            return value
        raise KeyError(key)

    def __setitem__(self, key: str, value: str):
        with self.conn:
            self.conn.execute(
                '''INSERT INTO KV (key, value) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value;''',
                (key, value),
            )

    def __delitem__(self, key: str):
        with self.conn:
            self.conn.execute('DELETE FROM KV WHERE key = ?;', (key,))

    def iter_prefix(self, prefix: str) -> Iterable[str]:
        cursor = self.conn.execute(
            'SELECT key FROM KV WHERE key LIKE ? ORDER BY key;', (prefix + '%',)
        )
        for row in cursor:
            yield row[0]

    def count_prefix(self, prefix: str) -> int:
        cursor = self.conn.execute(
            'SELECT COUNT(*) FROM KV WHERE key LIKE ?;', (prefix + '%',)
        )
        return cursor.fetchone()[0]

    def get_command(self, name: str) -> str | None:
        cursor = self.conn.execute('SELECT cmd FROM Command WHERE name = ?;', (name,))
        if row := cursor.fetchone():
            return row[0]
        return None

    def set_command(self, name: str, cmd: str):
        with self.conn:
            if cmd:
                self.conn.execute(
                    '''INSERT INTO Command (name, cmd) VALUES (?, ?)
                    ON CONFLICT(name) DO UPDATE SET cmd = excluded.cmd;''',
                    (name, cmd),
                )
            else:
                self.conn.execute('DELETE FROM Command WHERE name = ?;', (name,))

    def iter_commands(self) -> Iterable[str]:
        cursor = self.conn.execute('SELECT name FROM Command ORDER BY name;')
        for row in cursor:
            yield row[0]

    def save_media(
        self, chat_id: int, message_id: int, title: str, file_id: str
    ) -> bool:
        with self.conn:
            existing = self.conn.execute(
                'SELECT 1 FROM Media WHERE chat_id = ? AND message_id = ?;',
                (chat_id, message_id),
            ).fetchone()
            self.conn.execute(
                '''INSERT INTO Media (chat_id, message_id, title, file_id)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(chat_id, message_id)
                   DO UPDATE SET title = excluded.title, file_id = excluded.file_id;''',
                (chat_id, message_id, title, file_id),
            )
        return not existing

    def random_media(self) -> tuple[int, int] | None:
        cursor = self.conn.execute(
            'SELECT chat_id, message_id FROM Media ORDER BY RANDOM() LIMIT 1;'
        )
        if row := cursor.fetchone():
            return row[0], row[1]
        return None

    def random_media_file_id(self) -> str | None:
        cursor = self.conn.execute(
            'SELECT file_id FROM Media ORDER BY RANDOM() LIMIT 1;'
        )
        if row := cursor.fetchone():
            return row[0]
        return None

    def has_media(self, chat_id: int, message_id: int) -> bool:
        cursor = self.conn.execute(
            'SELECT 1 FROM Media WHERE chat_id = ? AND message_id = ?;',
            (chat_id, message_id),
        )
        return cursor.fetchone() is not None

    def iter_media(self) -> Iterable[tuple[int, int, str, str]]:
        yield from self.conn.execute(
            'SELECT chat_id, message_id, title, saved_at FROM Media ORDER BY id DESC;'
        )

    def delete_media(self, chat_id: int, message_id: int) -> str | None:
        with self.conn:
            cursor = self.conn.execute(
                'SELECT title FROM Media WHERE chat_id = ? AND message_id = ?;',
                (chat_id, message_id),
            )
            title_row = cursor.fetchone()
            if title_row is None:
                return None

            self.conn.execute(
                'DELETE FROM Media WHERE chat_id = ? AND message_id = ?;',
                (chat_id, message_id),
            )
        return title_row[0]


db = DataStore()
