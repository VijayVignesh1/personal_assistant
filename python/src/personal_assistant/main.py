from personal_assistant.application.app import Application
from personal_assistant.frontend.ui import UI


def main() -> None:
    application = Application()
    ui = UI(application)
    try:
        ui.launch()
    finally:
        application.close_episode()


if __name__ == "__main__":
    main()