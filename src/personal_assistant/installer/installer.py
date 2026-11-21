import argparse
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from personal_assistant.installer.env_analysis import analyze_environment
from personal_assistant.installer.services import (
    ensure_docker_installed,
    install_searxng,
)

VENV_PYTHON = Path(sys.executable)
VENV_DIR = VENV_PYTHON.parent.parent
UV_PATH = f"{VENV_DIR}/bin/uv"

PACKAGE_NAME = "personal-assistant"
PACKAGE_VERSION = version(PACKAGE_NAME)

def run(*args: str) -> None:
    command = [
        UV_PATH,
        *args,
        "--python",
        str(VENV_PYTHON),
    ]

    subprocess.run(
        command,
        check=True,
    )

def install( dev: bool) -> None:

    from personal_assistant.system.detector import detect_system_profile
    profile = detect_system_profile()
    report = analyze_environment(profile)

    if not report.supported:
        print("Errors:", report.errors)
        print("Warnings:", report.warnings)
        raise RuntimeError("Environment not supported")
    elif report.warnings:
        print("Warnings:", report.warnings)

    print("Environment is supported. Proceeding with installation...")

    # Install ML dependencies
    run("pip", "install", f"{PACKAGE_NAME}[ml]=={PACKAGE_VERSION}")

    # Install NVIDIA-specific dependencies
    if any(gpu.vendor == "NVIDIA" for gpu in profile.gpus):
        run("pip", "install", f"{PACKAGE_NAME}[nvidia]=={PACKAGE_VERSION}")

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
        run("pip", "install", f"{PACKAGE_NAME}[dev]=={PACKAGE_VERSION}")

    # Install docker automatically
    ensure_docker_installed()

    # Install SearXNG
    install_searxng()
    
    print("Installation completed successfully.")

def update(dev: bool = False) -> None:

    from personal_assistant.system.detector import detect_system_profile
    print("Checking system environment...")

    profile = detect_system_profile()
    report = analyze_environment(profile)

    if not report.supported:
        print("Errors:", report.errors)
        print("Warnings:", report.warnings)
        raise RuntimeError("Environment not supported")

    if report.warnings:
        print("Warnings:", report.warnings)

    print("Updating Personal Assistant...")

    # Upgrade core package and ML dependencies
    run(
        "python", "-m", "pip", "install",
        "--upgrade",
        f"{PACKAGE_NAME}[ml]=={PACKAGE_VERSION}"
    )

    # Upgrade NVIDIA dependencies
    if any(gpu.vendor == "NVIDIA" for gpu in profile.gpus):
        run(
            "python", "-m", "pip", "install",
            "--upgrade",
            f"{PACKAGE_NAME}[nvidia]=={PACKAGE_VERSION}"
        )

    # Upgrade PyTorch
    run(
        "python", "-m", "pip", "install",
        "--upgrade",
        "torch",
        "torchvision",
        "torchaudio",
        "--torch-backend=auto",
    )

    # Development dependencies
    if dev:
        run(
            "python", "-m", "pip", "install",
            "--upgrade",
            f"{PACKAGE_NAME}[dev]=={PACKAGE_VERSION}"
        )

    print("Personal Assistant updated successfully.")

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

    install(args.dev)


if __name__ == "__main__":
    main()
