import atexit
from sqlite3 import connect, Connection


class DataStore:
    def __init__(self):
        self.conn: Connection | None = None

    def connect(self, file: str):
        assert self.conn is None
        self.conn = connect(file, autocommit=False)
        atexit.register(self.conn.close)

        with self.conn:
            self.conn.executescript(
                '''CREATE TABLE IF NOT EXISTS Doc (
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
                '''
            )

    def close(self):
        c = self.conn
        self.conn = None
        (f := c.close)()
        atexit.unregister(f)

    def summary(self) -> str:
        n = self.conn.execute('SELECT COUNT(*) FROM Doc').fetchone()[0]
        m = self.conn.execute('SELECT COUNT(*) FROM KV').fetchone()[0]
        return f'{n} docs, {m} keys'

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


db = DataStore()
