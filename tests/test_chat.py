from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from personal_assistant import chat
from personal_assistant.application.app import Application


@pytest.fixture
def application(tmp_path):
    db_path = tmp_path / "test.db"
    return  Application(db_path=str(db_path))

@pytest.mark.asyncio
async def test_chat_streams_response():
    application = MagicMock()

    async def generate_response(message):
        assert message == "Hello"
        yield "Hello"
        yield " there!"

    application.generate_response = generate_response

    response = MagicMock()
    response.stream_token = AsyncMock()
    response.send = AsyncMock()

    incoming_message = MagicMock()
    incoming_message.content = "Hello"

    with (
        patch.object(
            chat.cl.user_session,
            "get",
            return_value=application,
        ),
        patch.object(
            chat.cl,
            "Message",
            return_value=response,
        ),
    ):
        await chat.chat(incoming_message)

    assert response.stream_token.await_count == 2

    response.stream_token.assert_any_await("Hello")
    response.stream_token.assert_any_await(" there!")
    response.send.assert_awaited_once()

@pytest.mark.asyncio
async def test_cleanup_closes_episode():
    application = MagicMock()
    application.close_episode = AsyncMock()

    with patch.object(
        chat.cl.user_session,
        "get",
        return_value=application,
    ):
        await chat.cleanup()

    application.close_episode.assert_awaited_once()