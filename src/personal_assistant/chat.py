import chainlit as cl

from personal_assistant.application.app import Application

# mypy: disable-error-code="no-untyped-call"

@cl.on_chat_start
async def start() -> None:
    application = Application(
        llm_model_name="gemma4:e2b",
        embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
    )

    cl.user_session.set("application", application)


@cl.on_message
async def chat(message: cl.Message) -> None:
    application: Application = cl.user_session.get("application")

    response = cl.Message(content="")

    async for chunk in application.generate_response(message.content):
        await response.stream_token(chunk)

    await response.send()


@cl.on_chat_end
async def cleanup() -> None:
    application = cl.user_session.get("application")

    if application is not None:
        await application.close_episode()