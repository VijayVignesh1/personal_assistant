import datetime
import sqlite3
from pathlib import Path

from personal_assistant.objects.episode import Episode
from personal_assistant.objects.memory import Memory
from personal_assistant.objects.message import Message
from personal_assistant.objects.retrieved_context import RetrievedContext


class Database:
    """
    A class to manage the SQLite database for the personal assistant application.
    """
    def __init__(self, db_path: str | Path):
        self.db_path = db_path
        self.connection: sqlite3.Connection | None = None
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
        assert db is not None, "Database connection is not established."
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

            db.execute('''CREATE TABLE IF NOT EXISTS retrieved_context (
                id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL REFERENCES message(message_id),
                memory_id TEXT REFERENCES memory(memory_id),
                retrieved_content TEXT NOT NULL,
                similarity_score REAL,
                retrieval_rank INTEGER,
                retrieval_timestamp TEXT NOT NULL
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
        assert db is not None, "Database connection is not established."
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
        assert db is not None, "Database connection is not established."
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
        assert db is not None, "Database connection is not established."
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
        finally:
            self._close()

    def get_messages_by_episode(self, episode_id: str) -> list[Message]:
        """
        Retrieve all messages associated with a specific episode ID.
        :param episode_id: The ID of the episode for which to retrieve messages.
        :return: A list of Message instances associated with the episode.
        """
        if not self.connection:
            self._connect()
        db = self.connection
        assert db is not None, "Database connection is not established."
        cursor = db.execute('''
            SELECT *
            FROM message
            WHERE episode_id = ?
            ORDER BY timestamp ASC;
        ''', (episode_id,))
        messages = [Message(message_id=row[0], episode_id=row[1], timestamp=row[2], role=row[3], content=row[4]) for row in cursor.fetchall()]
        self._close()
        return messages

    def get_full_episode(self, episode_id: str) -> tuple[Episode | None, list[Message]]:
        """
        Retrieve an episode and its associated messages by episode ID.
        :param episode_id: The ID of the episode to retrieve.
        :return: A tuple containing the Episode instance and a list of associated Message instances.
        """
        episode = self.get_episode(episode_id)
        if episode is None:
            return None, []
        episode_id, _, _ = episode.episode_id, episode.started_at, episode.ended_at
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
        assert db is not None, "Database connection is not established."
        end_time = datetime.datetime.now(tz=datetime.UTC).isoformat()
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

    def save_memory(self, memories: list[Memory]) -> None:
        """
        Save a list of Memory instances to the database.
        :param memories: A list of Memory instances to save.
        """
        if not self.connection:
            self._connect()
        db = self.connection
        assert db is not None, "Database connection is not established."
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

    def get_all_memories(self) -> list[Memory]:
        """
        Retrieve all memories from the database.
        :return: A list of Memory instances.
        """
        if not self.connection:
            self._connect()
        db = self.connection
        assert db is not None, "Database connection is not established."
        try:
            cursor = db.execute('SELECT content, timestamp, importance, episode_id, memory_id, embedding, model_name FROM memory')
            rows = cursor.fetchall()
            memories = [Memory(content=row[0], timestamp=row[1], importance=row[2], episode_id=row[3], memory_id=row[4], embedding=row[5], model_name=row[6]) for row in rows]
            return memories
        finally:
            self._close()

    def save_retrieved_context(self, retrieved_contexts: list[RetrievedContext]) -> None:
        """
        Save a list of RetrievedContext instances to the database.
        :param retrieved_contexts: A list of RetrievedContext instances to save.
        """
        if not self.connection:
            self._connect()
        db = self.connection
        assert db is not None, "Database connection is not established."
        try:
            for context in retrieved_contexts:
                db.execute('''
                    INSERT INTO retrieved_context (id, message_id, memory_id, retrieved_content, similarity_score, retrieval_rank, retrieval_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (context.id, context.message_id, context.memory_id, context.retrieved_content, context.similarity_score, context.retrieval_rank, context.retrieval_timestamp))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            self._close()

    def get_all_retrieved_contexts(self, message_id) -> list[RetrievedContext]:
        """
        Retrieve all retrieved contexts for a specific message ID from the database.
        :param message_id: The ID of the message to retrieve contexts for.
        :return: A list of RetrievedContext instances.
        """
        if not self.connection:
            self._connect()
        db = self.connection
        assert db is not None, "Database connection is not established."
        try:
            cursor = db.execute('SELECT id, message_id, memory_id, retrieved_content, similarity_score, retrieval_rank, retrieval_timestamp FROM retrieved_context WHERE message_id = ?', (message_id,))
            rows = cursor.fetchall()
            contexts = [RetrievedContext(id=row[0], message_id=row[1], memory_id=row[2], retrieved_content=row[3], similarity_score=row[4], retrieval_rank=row[5], retrieval_timestamp=row[6]) for row in rows]
            return contexts
        finally:
            self._close()