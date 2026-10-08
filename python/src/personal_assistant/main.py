from personal_assistant.application.app import Application
from personal_assistant.frontend.ui import UI


def main() -> None:
    application = Application(model_name = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit")
    ui = UI(application)
    try:
        ui.launch()
    finally:
        application.close_episode()


if __name__ == "__main__":
    main()