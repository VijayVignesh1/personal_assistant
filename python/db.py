import sqlite3
import uuid
from datetime import datetime

class Database:
    def __init__(self, db_path='../database/diary.db'):
        self.db_path = db_path
        self.connection = None
            
    def _connect(self):
        self.connection = sqlite3.connect(self.db_path)
        self.connection.execute("PRAGMA foreign_keys = ON")

    def _close(self):
        if self.connection:
            self.connection.close()
            self.connection = None

    def initialize_db(self):
        self._connect()
        db = self.connection
        try:
            db.execute('''CREATE TABLE IF NOT EXISTS episode (
                episode_id TEXT NOT NULL PRIMARY KEY,
                started_at TEXT NOT NULL,
                ended_at TEXT
            )''')

            db.execute('''CREATE TABLE IF NOT EXISTS message (
                message_id TEXT PRIMARY KEY,
                episode_id TEXT NOT NULL REFERENCES episode (episode_id),
                timestamp TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL
            )''')

            db.execute('''CREATE TABLE IF NOT EXISTS memory (
                memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                importance REAL NOT NULL,
                embedding BLOB,
                model_name TEXT,
                episode_id TEXT NOT NULL REFERENCES episode (episode_id)
            )''')

            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            self._close()

    def create_episode(self):
        episode_id = str(uuid.uuid4())
        started_at = datetime.now().isoformat()
        if not self.connection:
            self._connect()
        db = self.connection
        try:
            db.execute("INSERT INTO episode(episode_id, started_at) VALUES (?, ?)", (episode_id, started_at))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            self._close()
        return episode_id

    def add_message(self, episode_id, role, content):
        message_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        if not self.connection:
            self._connect()
        db = self.connection
        try:
            db.execute('''
                INSERT INTO message (
                    message_id,
                    episode_id,
                    timestamp,
                    role,
                    content
                )
                VALUES (?, ?, ?, ?, ?);
            ''', (message_id, episode_id, timestamp, role, content))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            self._close()
        return message_id

    def get_episode(self, episode_id):
        if not self.connection:
            self._connect()
        db = self.connection
        try:
            cursor = db.execute('''
                SELECT episode_id, started_at, ended_at
                FROM episode
                WHERE episode_id = ?;
            ''', (episode_id,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            episode_id, started_at, ended_at = row
            return {
                "episode_id": episode_id,
                "started_at": started_at,
                "ended_at": ended_at
            }
        except Exception:
            raise
        finally:
            self._close()

    def get_messages_by_episode(self, episode_id):
        if not self.connection:
            self._connect()
        db = self.connection
        cursor = db.execute('''
            SELECT *
            FROM message
            WHERE episode_id = ?
            ORDER BY timestamp ASC;
        ''', (episode_id,))
        messages = cursor.fetchall()
        self._close()
        return messages

    def get_full_episode(self, episode_id):
        episode = self.get_episode(episode_id)
        if episode is None:
            return None
        episode_id, started_at, ended_at = episode["episode_id"], episode["started_at"], episode["ended_at"]
        messages = self.get_messages_by_episode(episode_id)
        return {
            "episode_id": episode_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "messages": messages
        }

    def close_episode(self, episode_id):
        if not self.connection:
            self._connect()
        db = self.connection
        end_time = datetime.now().isoformat()
        try:
            cursor = db.execute('''UPDATE episode
            SET ended_at = ?
            WHERE episode_id = ?
            AND ended_at IS NULL;''', (end_time, episode_id))
            db.commit()
            if cursor.rowcount == 0:
                raise ValueError(f"Episode {episode_id} is already closed or does not exist.")
        except Exception:
            db.rollback()
            raise
        finally:
            self._close()

    def save_memory(self, memories):
        if not self.connection:
            self._connect()
        db = self.connection
        try:
            for memory in memories:
                db.execute('''
                    INSERT INTO memory (content, timestamp, importance, episode_id)
                    VALUES (?, ?, ?, ?)
                ''', (memory["content"], memory["timestamp"], memory["importance"], memory["episode_id"]))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            self._close()