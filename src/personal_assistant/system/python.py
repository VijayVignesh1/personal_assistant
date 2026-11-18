import os
import platform
import subprocess
from pathlib import Path


def find_python_versions() -> list[dict[str, str]]:
    system = platform.system()

    search_dirs = {
        "Linux": [
            "/usr/bin",
            "/usr/local/bin",
            "/bin",
            "/opt",
            "/usr/local/opt",
            str(Path.home() / ".pyenv/versions"),
            str(Path.home() / ".local/bin"),
        ],
        "Darwin": [
            "/usr/bin",
            "/usr/local/bin",
            "/opt/homebrew/bin",
            "/opt/local/bin",
            "/Library/Frameworks/Python.framework/Versions",
            str(Path.home() / ".pyenv/versions"),
            str(Path.home() / ".local/bin"),
        ],
    }

    if system not in search_dirs:
        raise OSError(f"Unsupported platform: {system}")

    executables = set()

    # Search PATH
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        path = Path(directory)

        if path.is_dir():
            for executable in path.glob("python*"):
                if executable.is_file() and os.access(executable, os.X_OK):
                    executables.add(executable.resolve())

    # Search common installation directories
    for directory in search_dirs[system]:
        path = Path(directory)

        if not path.is_dir():
            continue

        for executable in path.glob("python*"):
            if executable.is_file() and os.access(executable, os.X_OK):
                executables.add(executable.resolve())

    # Verify interpreters
    versions = {}

    for executable in executables:
        try:
            result = subprocess.run(
                [
                    str(executable),
                    "-c",
                    "import sys; print('.'.join(map(str, sys.version_info[:3])))"
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=False
            )

            if result.returncode == 0:
                versions[str(executable)] = result.stdout.strip()

        except (OSError, subprocess.TimeoutExpired):
            continue

    return [
        {
            version: executable,
        }
        for executable, version in sorted(
            versions.items(),
            key=lambda item: tuple(map(int, item[1].split("."))),
        )
    ]


if __name__ == "__main__":
    print(find_python_versions())