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
            self.conn.execute(
                '''CREATE TABLE IF NOT EXISTS Doc (
                    id INTEGER PRIMARY KEY NOT NULL,
                    name TEXT UNIQUE NOT NULL,
                    text TEXT NOT NULL
                )'''
            )

    def close(self):
        c = self.conn
        self.conn = None
        (f := c.close)()
        atexit.unregister(f)

    def summary(self) -> str:
        n = self.conn.execute('SELECT COUNT(*) FROM Doc').fetchone()[0]
        return f'{n} docs'

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


db = DataStore()
