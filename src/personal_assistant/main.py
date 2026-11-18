import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Launch the Chainlit application."""

    app_file = Path(__file__).parent / "chat.py"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "chainlit",
            "run",
            str(app_file),
        ],
        check=True,
    )