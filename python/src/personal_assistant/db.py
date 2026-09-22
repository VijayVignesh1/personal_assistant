import sqlite3
from typing import List
from datetime import datetime
from personal_assistant.objects.memory import Memory
from personal_assistant.objects.episode import Episode
from personal_assistant.objects.message import Message

class Database:
    """
    A class to manage the SQLite database for the personal assistant application.
    """
    def __init__(self, db_path='../database/diary.db'):
        self.db_path = db_path
        self.connection = None
        self._initialize_db()
            
    def _connect(self) -> None:
        """
        Establish a connection to the SQLite database and enable foreign key support.
        """
        self.connection = sqlite3.connect(self.db_path)
        self.connection.execute("PRAGMA foreign_keys = ON")

    def _close(self) -> None:
        """
        Close the database connection if it is open.
        """
        if self.connection:
            self.connection.close()
            self.connection = None

    def _initialize_db(self) -> None:
        """
        Initialize the database by creating the necessary tables if they do not already exist.
        """
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
                memory_id TEXT PRIMARY KEY,
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

    def create_episode(self, episode: Episode) -> str:
        """
        Create a new episode in the database.
        :param episode: An instance of the Episode class containing episode details.
        :return: The ID of the created episode.
        """
        if not self.connection:
            self._connect()
        db = self.connection
        try:
            db.execute("INSERT INTO episode(episode_id, started_at) VALUES (?, ?)", (episode.episode_id, episode.started_at))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            self._close()
        return episode.episode_id

    def add_message(self, message: Message) -> str:
        """
        Add a new message to the database.
        :param message: An instance of the Message class containing message details.
        :return: The ID of the created message.
        """
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
            ''', (message.message_id, message.episode_id, message.timestamp, message.role, message.content))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            self._close()
        return message.message_id

    def get_episode(self, episode_id: str) -> Episode | None:
        """
        Retrieve an episode from the database by its ID.
        :param episode_id: The ID of the episode to retrieve.
        :return: An instance of the Episode class if found, otherwise None.
        """
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
            episode = Episode(episode_id=row[0], started_at=row[1], ended_at=row[2])
            
            return episode
        except Exception:
            raise
        finally:
            self._close()

    def get_messages_by_episode(self, episode_id: str) -> List[Message]:
        """
        Retrieve all messages associated with a specific episode ID.
        :param episode_id: The ID of the episode for which to retrieve messages.
        :return: A list of Message instances associated with the episode.
        """
        if not self.connection:
            self._connect()
        db = self.connection
        cursor = db.execute('''
            SELECT *
            FROM message
            WHERE episode_id = ?
            ORDER BY timestamp ASC;
        ''', (episode_id,))
        messages = [Message(message_id=row[0], episode_id=row[1], timestamp=row[2], role=row[3], content=row[4]) for row in cursor.fetchall()]
        self._close()
        return messages

    def get_full_episode(self, episode_id: str) -> tuple[Episode, List[Message]]:
        """
        Retrieve an episode and its associated messages by episode ID.
        :param episode_id: The ID of the episode to retrieve.
        :return: A tuple containing the Episode instance and a list of associated Message instances.
        """
        episode = self.get_episode(episode_id)
        if episode is None:
            return None, []
        episode_id, started_at, ended_at = episode.episode_id, episode.started_at, episode.ended_at
        messages = self.get_messages_by_episode(episode_id)
        return episode, messages

    def close_episode(self, episode_id: str) -> None:
        """
        Close an episode by setting its ended_at timestamp to the current time.
        :param episode_id: The ID of the episode to close.
        :raises ValueError: If the episode is already closed or does not exist.
        """
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

    def save_memory(self, memories: List[Memory]) -> None:
        """
        Save a list of Memory instances to the database.
        :param memories: A list of Memory instances to save.
        """
        if not self.connection:
            self._connect()
        db = self.connection
        try:
            for memory in memories:
                db.execute('''
                    INSERT INTO memory (content, timestamp, importance, episode_id, memory_id, embedding, model_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (memory.content, memory.timestamp, memory.importance, memory.episode_id, memory.memory_id, memory.embedding, memory.model_name))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            self._close()