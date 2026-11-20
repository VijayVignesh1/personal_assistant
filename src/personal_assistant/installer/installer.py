import argparse
import os
import pathlib
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from personal_assistant.installer.env_analysis import analyze_environment
from personal_assistant.installer.services import ensure_uv, install_searxng, ensure_docker_installed

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VENV_DIR = PROJECT_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
UV_PATH = ensure_uv()

def run(*args: str) -> None:
    command = [
        UV_PATH,
        *args,
        "--python",
        str(VENV_PYTHON),
    ]

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )

def install(tomlconfig: dict[str, Any], dev: bool) -> None:

    # Create virtual environment
    run("venv")

    # install psutil and py-cpuinfo
    run("pip", "install", "psutil")
    run("pip", "install", "py-cpuinfo")

    from personal_assistant.system.detector import detect_system_profile
    profile = detect_system_profile()
    report = analyze_environment(profile, tomlconfig)

    if not report.supported:
        print("Errors:", report.errors)
        print("Warnings:", report.warnings)
        raise RuntimeError("Environment not supported")
    elif report.warnings:
        print("Warnings:", report.warnings)

    print("Environment is supported. Proceeding with installation...")

    # Install base dependencies
    run("pip", "install", "-e", str(PROJECT_ROOT))
    # Install ML dependencies
    run("pip", "install", "-e", f"{PROJECT_ROOT}[ml]")

    # Install NVIDIA-specific dependencies
    if any(gpu.vendor == "NVIDIA" for gpu in profile.gpus):
        run("pip", "install", "-e", f"{PROJECT_ROOT}[nvidia]")

    # install torch with auto backend
    run(
        "pip", "install",
        "torch",
        "torchvision",
        "torchaudio",
        "--torch-backend=auto",
    )
    # Install development dependencies if requested
    if dev:
        run("pip", "install", "--group", "dev")

    # Install docker automatically
    ensure_docker_installed()

    # Install SearXNG
    install_searxng()
    
    print("Installation completed successfully.")

def update() -> None:
    run(
        "pip", "install",
        "--upgrade",
        "-e", str(PROJECT_ROOT),
    )
    print("Dependencies updated successfully.")

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Personal Assistant Installer"
    )

    parser.add_argument(
        "--dev",
        action="store_true",
        help="Install development dependencies",
    )

    args = parser.parse_args()

    # Check root privileges
    if os.geteuid() != 0:
        result = subprocess.run(
            ["sudo", sys.executable, *sys.argv],
            check=False
        )

        if result.returncode != 0:
            print(f"Installation failed with exit code: {result.returncode}")
            sys.exit(result.returncode)

        return

    PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
    pyproject_path = PROJECT_ROOT / "pyproject.toml"

    if not pyproject_path.exists():
        print("pyproject.toml not found")
        sys.exit(1)

    with pyproject_path.open("rb") as f:
        tomlconfig = tomllib.load(f)

    install(tomlconfig, args.dev)


if __name__ == "__main__":
    main()
