import platform
import secrets
import shutil
import subprocess
import time
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


def ensure_docker_installed() -> None:
    """Install and start Docker on macOS or Linux."""

    system = platform.system().lower()

    if not shutil.which("docker"):

        print("Docker not found. Installing...")

        if system == "darwin":

            if not shutil.which("brew"):
                raise RuntimeError(
                    "Please install Homebrew first: https://brew.sh/"
                )

            subprocess.run(
                ["brew", "install", "--cask", "docker"],
                check=True
            )

        elif system == "linux":

            if shutil.which("apt-get"):
                subprocess.run(
                    ["sudo", "apt-get", "update"],
                    check=True
                )
                subprocess.run(
                    ["sudo", "apt-get", "install", "-y", "docker.io"],
                    check=True
                )

            elif shutil.which("dnf"):
                subprocess.run(
                    ["sudo", "dnf", "install", "-y", "docker"],
                    check=True
                )

            else:
                raise RuntimeError(
                    "Unsupported Linux package manager."
                )

        else:
            raise RuntimeError(f"Unsupported OS: {system}")

    # Start Docker
    if system == "darwin":
        subprocess.run(["open", "-a", "Docker"], check=True)

    elif system == "linux":
        subprocess.run(
            ["sudo", "systemctl", "enable", "--now", "docker"],
            check=True
        )

    # Wait for Docker daemon
    for _ in range(60):
        result = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )

        if result.returncode == 0:
            print("Docker is ready!")
            return

        time.sleep(2)

    raise RuntimeError("Docker failed to start.")

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