
import asyncio

from personal_assistant.application.app import Application
from personal_assistant.frontend.ui import UI


def main() -> None:
    application = Application(llm_model_name = "gemma4:e2b",
                              embedding_model_name = "Qwen/Qwen3-Embedding-0.6B")
    ui = UI(application)
    try:
        ui.launch()
    finally:
        asyncio.run(application.close_episode())


if __name__ == "__main__":
    main()