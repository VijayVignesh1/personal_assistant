import secrets
import shutil
import subprocess
from pathlib import Path

import yaml

SEARXNG_CONTAINER = "searxng"
SEARXNG_IMAGE = "searxng/searxng:latest"
SEARXNG_PORT = 8080

SEARXNG_DIR = Path.home() / ".config" / "diary-ai" / "searxng"
SEARXNG_SETTINGS = SEARXNG_DIR / "settings.yml"

def configure_searxng() -> None:
    """Create or update SearXNG configuration."""

    SEARXNG_DIR.mkdir(parents=True, exist_ok=True)

    if SEARXNG_SETTINGS.exists():
        with SEARXNG_SETTINGS.open("r") as f:
            settings = yaml.safe_load(f) or {}
    else:
        settings = {}

    settings["use_default_settings"] = True

    server = settings.setdefault("server", {})

    server.setdefault(
        "secret_key",
        secrets.token_urlsafe(32)
    )

    server["limiter"] = False
    server["image_proxy"] = True

    search = settings.setdefault("search", {})

    search["formats"] = ["html", "json"]

    with SEARXNG_SETTINGS.open("w") as f:
        yaml.safe_dump(
            settings,
            f,
            sort_keys=False
        )

    print("SearXNG configuration updated.")
    print("JSON API enabled.")

def ensure_uv() -> str:
    """Find uv or install it if missing."""

    import os
    import pwd
    import shutil
    import subprocess
    from pathlib import Path

    # Check PATH first
    uv_path = shutil.which("uv")

    if uv_path:
        print(f"uv is already installed: {uv_path}")
        return uv_path

    # Detect original user
    sudo_user = os.environ.get("SUDO_USER")

    if sudo_user:
        user_home = Path(pwd.getpwnam(sudo_user).pw_dir)
    else:
        user_home = Path.home()

    # Check common installation locations
    possible_paths = [
        user_home / "miniconda3/bin/uv",
        user_home / ".local/bin/uv",
        Path("/root/.local/bin/uv"),
        Path("/usr/local/bin/uv"),
    ]

    for path in possible_paths:
        if path.is_file():
            print(f"uv is already installed: {path}")
            return str(path)

    # Install uv only if genuinely missing
    print("uv not found. Installing uv...")

    subprocess.run(
        [
            "bash",
            "-c",
            "curl -LsSf https://astral.sh/uv/install.sh | sh"
        ],
        check=True,
    )

    possible_paths = [
        Path.home() / ".local/bin/uv",
        user_home / ".local/bin/uv",
        Path("/root/.local/bin/uv"),
    ]

    for path in possible_paths:
        if path.is_file():
            print(f"uv installed successfully: {path}")
            return str(path)

    raise RuntimeError("uv installation failed.")

def install_searxng() -> None:
    """Install and start SearXNG using Docker."""

    configure_searxng()

    # Check Docker availability
    if shutil.which("docker") is None:
        raise RuntimeError(
            "Docker is not installed. Install Docker first: "
            "https://docs.docker.com/get-docker/"
        )

    # Check Docker daemon
    try:
        subprocess.run(
            ["docker", "info"],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        raise RuntimeError(
            "Docker daemon is not running or the current user "
            "does not have permission to access Docker."
        )

    # Create configuration directory
    SEARXNG_DIR.mkdir(parents=True, exist_ok=True)

    # Check whether container exists
    existing = subprocess.run(
        [
            "docker", "ps", "-aq",
            "--filter", f"name=^{SEARXNG_CONTAINER}$",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    if existing.stdout.strip():

        # Check whether container is running
        running = subprocess.run(
            [
                "docker", "ps", "-q",
                "--filter", f"name=^{SEARXNG_CONTAINER}$",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        if running.stdout.strip():
            print("SearXNG is already running. Restarting...")

            subprocess.run(
                ["docker", "restart", SEARXNG_CONTAINER],
                check=True,
            )

        else:
            print("Starting existing SearXNG container...")

            subprocess.run(
                ["docker", "start", SEARXNG_CONTAINER],
                check=True,
            )

    else:
        print("Creating SearXNG container...")

        subprocess.run(
            [
                "docker", "run",
                "-d",
                "--name", SEARXNG_CONTAINER,
                "--restart", "unless-stopped",
                "-p", f"{SEARXNG_PORT}:8080",
                "-v",
                f"{SEARXNG_SETTINGS}:/etc/searxng/settings.yml:ro",
                SEARXNG_IMAGE,
            ],
            check=True,
        )

    print("SearXNG is ready!")
    print(f"URL: http://localhost:{SEARXNG_PORT}")

    print("SearXNG installed successfully!")
    print(f"URL: http://localhost:{SEARXNG_PORT}")