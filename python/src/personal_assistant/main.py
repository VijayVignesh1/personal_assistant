import chainlit as cl

from personal_assistant.application.app import Application


@cl.on_chat_start
async def start() -> None:
    """Initialize the application for this chat session."""
    application = Application(
        llm_model_name="gemma4:e2b",
        embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
    )

    cl.user_session.set("application", application)  # type: ignore[no-untyped-call]


@cl.on_message
async def chat(message: cl.Message) -> None:
    """Stream the application's response to the chat UI."""
    application: Application = cl.user_session.get("application")  # type: ignore[no-untyped-call]

    response = cl.Message(content="")

    async for chunk in application.generate_response(message.content):
        await response.stream_token(chunk)

    await response.send()  # type: ignore[no-untyped-call]


@cl.on_chat_end
async def cleanup() -> None:
    """Clean up application resources when the chat ends."""
    application = cl.user_session.get("application")  # type: ignore[no-untyped-call]

    if application is not None:
        await application.close_episode()
