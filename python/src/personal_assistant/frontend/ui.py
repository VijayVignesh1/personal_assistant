from collections.abc import AsyncGenerator
from typing import Any

import gradio as gr

from personal_assistant.application.app import Application


class UI:
    def __init__(self, application: Application) -> None:
        """Initialize the UI class."""
        self.chat_interface = gr.ChatInterface(
            fn=self._chat
        )
        self.application = application

    def launch(self) -> None:
        """Launch the Gradio interface."""
        self.chat_interface.launch()
        
    async def _chat(self, message: str, history: list[dict[str, Any]]) -> AsyncGenerator[str, None]:
        """Display a message to the user."""
        async for chunk in self.application.generate_response(message):
            yield chunk
