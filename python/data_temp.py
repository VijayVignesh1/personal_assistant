import sqlite3
import uuid
from datetime import datetime
from llm.memory_generator import dummy_llm_memory_extraction

def connect_db(db_path='../database/diary.db'):
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA foreign_keys = ON")
    return db

def initialize_db(db_path='../database/diary.db'):
    db = connect_db(db_path=db_path)
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
            episode_id TEXT NOT NULL REFERENCES episode (episode_id)
        )''')

        db.commit()
    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()

# push data to the database
def create_episode(db_path='../database/diary.db'):
    episode_id = str(uuid.uuid4())
    started_at = datetime.now().isoformat()
    db = connect_db(db_path=db_path)
    try:
        db.execute("INSERT INTO episode(episode_id, started_at) VALUES (?, ?)", (episode_id, started_at))
        db.commit()
    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()
    return episode_id


# insert message
def add_message(episode_id, role, content, db_path='../database/diary.db'):
    message_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    db = connect_db(db_path=db_path)
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
    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()
    return message_id

# retrieve episode by id
def get_episode(episode_id, db_path='../database/diary.db'):
    db = connect_db(db_path=db_path)
    cursor = db.execute('''
        SELECT episode_id, started_at, ended_at
        FROM episode
        WHERE episode_id = ?;
    ''', (episode_id,))
    row = cursor.fetchone()
    db.close()
    if row is None:
        return None
    
    episode_id, started_at, ended_at = row
    return {
        "episode_id": episode_id,
        "started_at": started_at,
        "ended_at": ended_at
    }

# retrieve messages from episode id
def get_messages_by_episode(episode_id, db_path='../database/diary.db'):
    db = connect_db(db_path=db_path)
    cursor = db.execute('''
        SELECT *
        FROM message
        WHERE episode_id = ?
        ORDER BY timestamp ASC;
    ''', (episode_id,))
    messages = cursor.fetchall()
    db.close()
    return messages


def get_full_episode(episode_id, db_path='../database/diary.db'):
    episode = get_episode(episode_id, db_path=db_path)
    if episode is None:
        return None
    episode_id, started_at, ended_at = episode["episode_id"], episode["started_at"], episode["ended_at"]
    messages = get_messages_by_episode(episode_id, db_path=db_path)
    return {
        "episode_id": episode_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "messages": messages
    }

# end episode
def close_episode(episode_id, db_path='../database/diary.db'):
    db = connect_db(db_path=db_path)
    end_time = datetime.now().isoformat()
    try:
        cursor = db.execute('''UPDATE episode
        SET ended_at = ?
        WHERE episode_id = ?
        AND ended_at IS NULL;''', (end_time, episode_id))
        db.commit()
        if cursor.rowcount == 0:
            raise ValueError(f"Episode {episode_id} is already closed or does not exist.")
    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()

def memory_extractor(episode_id, db_path='../database/diary.db'):
    messages = get_messages_by_episode(episode_id, db_path=db_path)
    messages = ["\n".join(msg) for msg in messages]

    # send messages to LLM for memory extraction
    memories = []
    prospect_mems = dummy_llm_memory_extraction(messages)
    for mem in prospect_mems:
        memories.append({
            "content": mem["content"],
            "timestamp": datetime.now().isoformat(),
            "importance": mem["importance"],
            "episode_id": episode_id
        })
    return memories

def save_memory(episode_id, db_path='../database/diary.db'):
    memories = memory_extractor(episode_id, db_path=db_path)
    for memory in memories:
        if memory["importance"] < 0.7:
            print(f"Memory importance {memory['importance']} is below threshold. Not saving.")
            return
        
        db = sqlite3.connect(db_path)
        try:
            db.execute('''
                INSERT INTO memory (content, timestamp, importance, episode_id)
                VALUES (?, ?, ?, ?)
            ''', (memory["content"], memory["timestamp"], memory["importance"], memory["episode_id"]))
            db.commit()
        except Exception as e:
            print(f"Error saving memory: {e}")
            db.rollback()
            raise
        finally:
            db.close()