import sqlite3
from python.data_temp import (
    initialize_db,
    create_episode,
    add_message,
    get_episode,
    get_messages_by_episode,
    get_full_episode,
    close_episode,
    memory_extractor,
    save_memory,
)


def test_create_episode(db_path='../database/test.db'):
    episode_id = create_episode(db_path=db_path)

    assert episode_id is not None
    assert isinstance(episode_id, str)

    episode = get_episode(episode_id, db_path=db_path)

    assert episode is not None
    assert episode["episode_id"] == episode_id
    assert episode["started_at"] is not None
    assert episode["ended_at"] is None

    print("PASS: create_episode")


def test_add_and_get_messages(db_path='../database/test.db'):
    episode_id = create_episode(db_path=db_path)

    message_1 = add_message(
        episode_id,
        "user",
        "Hello",
        db_path=db_path
    )

    message_2 = add_message(
        episode_id,
        "assistant",
        "Hi! How can I help?",
        db_path=db_path
    )

    messages = get_messages_by_episode(episode_id, db_path=db_path)

    assert len(messages) == 2
    assert messages[0][0] == message_1
    assert messages[0][3] == "user"
    assert messages[0][4] == "Hello"

    assert messages[1][0] == message_2
    assert messages[1][3] == "assistant"
    assert messages[1][4] == "Hi! How can I help?"

    print("PASS: add_message + get_messages_by_episode")


def test_full_episode(db_path='../database/test.db'):
    episode_id = create_episode(db_path=db_path)

    add_message(
        episode_id,
        "user",
        "What is machine learning?",
        db_path=db_path
    )

    add_message(
        episode_id,
        "assistant",
        "Machine learning is...",
        db_path=db_path
    )

    episode = get_full_episode(episode_id, db_path=db_path)

    assert episode is not None
    assert episode["episode_id"] == episode_id
    assert episode["started_at"] is not None
    assert episode["ended_at"] is None
    assert len(episode["messages"]) == 2

    print("PASS: get_full_episode")


def test_close_episode(db_path='../database/test.db'):
    episode_id = create_episode(db_path=db_path)

    episode_before = get_episode(episode_id, db_path=db_path)

    assert episode_before["ended_at"] is None

    close_episode(episode_id, db_path=db_path)

    episode_after = get_episode(episode_id, db_path=db_path)

    assert episode_after is not None
    assert episode_after["ended_at"] is not None

    print("PASS: close_episode")


def test_close_episode_twice(db_path='../database/test.db'):
    episode_id = create_episode(db_path=db_path)

    close_episode(episode_id, db_path=db_path)

    try:
        close_episode(episode_id, db_path=db_path)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    print("PASS: cannot close episode twice")


def test_nonexistent_episode(db_path='../database/test.db'):
    fake_episode_id = "does-not-exist"

    episode = get_episode(fake_episode_id, db_path=db_path)

    assert episode is None

    full_episode = get_full_episode(fake_episode_id, db_path=db_path)

    assert full_episode is None

    print("PASS: nonexistent episode")


def test_invalid_role(db_path='../database/test.db'):
    episode_id = create_episode(db_path=db_path)

    try:
        add_message(
            episode_id,
            "banana",
            "This should fail"
        )
        assert False, "Expected database constraint error"
    except Exception:
        pass

    print("PASS: invalid role rejected")


def test_invalid_episode_id(db_path='../database/test.db'):
    fake_episode_id = "does-not-exist"

    try:
        add_message(
            fake_episode_id,
            "user",
            "This should fail"
        )
        assert False, "Expected foreign key constraint error"
    except Exception:
        pass

    print("PASS: invalid episode_id rejected")

def test_memory_extractor(db_path='../database/test.db'):

    episode_id = create_episode(db_path=db_path)

    _ = add_message(
        episode_id,
        "user",
        "Hello",
        db_path=db_path
    )

    memory = memory_extractor(
        episode_id,
        db_path=db_path
    )

    assert type(memory[0]["content"]) == str
    assert memory[0]["content"] is not None
    assert memory[0]["episode_id"] == episode_id
    assert memory[0]["importance"] == 0.8
    assert "timestamp" in memory[0]
    assert memory[0]["timestamp"] is not None

    print("PASS: memory_extractor")


def test_save_memory(db_path='../database/test.db'):

    episode_id = create_episode(db_path=db_path)

    _ = add_message(
        episode_id,
        "user",
        "Hello",
        db_path=db_path
    )

    save_memory(
        episode_id,
        db_path=db_path
    )

    db = sqlite3.connect(db_path)

    rows = db.execute("""
        SELECT content, timestamp, importance, episode_id
        FROM memory
        WHERE episode_id = ?
    """, (episode_id,)).fetchall()

    db.close()

    assert rows is not None
    assert type(rows[0][0]) == str
    assert rows[0][0] is not None
    assert rows[0][2] == 0.8
    assert rows[0][3] == episode_id

    print("PASS: save_memory")

    
def run_all_tests():
    initialize_db(db_path='../database/test.db')

    test_create_episode(db_path='../database/test.db')
    test_add_and_get_messages(db_path='../database/test.db')
    test_full_episode(db_path='../database/test.db')
    test_close_episode(db_path='../database/test.db')
    test_close_episode_twice(db_path='../database/test.db')
    test_nonexistent_episode(db_path='../database/test.db')
    test_invalid_role(db_path='../database/test.db')
    test_invalid_episode_id(db_path='../database/test.db')

    test_memory_extractor(db_path='../database/test.db')
    test_save_memory(db_path='../database/test.db')
    print("\nAll tests passed!")


if __name__ == "__main__":
    run_all_tests()
