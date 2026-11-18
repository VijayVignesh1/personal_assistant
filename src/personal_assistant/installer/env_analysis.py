import tomllib
from typing import Any

from personal_assistant.installer.schema import EnvironmentReport
from personal_assistant.system.schema import SystemProfile


def analyze_environment(
    profile: SystemProfile,
    tomlconfig: dict[str, Any]
) -> EnvironmentReport:

    project = tomlconfig["project"]
    requirements = tomlconfig["tool"]["personal-assistant"]["requirements"]

    python_req = project["requires-python"]
    min_python = python_req.split(",")[0].replace(">=", "")  # ">=3.11,<3.14"
    max_python = python_req.split(",")[1].replace("<", "")  # ">=3.11,<3.14"
    min_ram = requirements["minimum-ram"].replace("GB", "") 
    recommended_ram = requirements["recommended-ram"].replace("GB", "")
    min_storage = requirements["minimum-storage"].replace("GB", "") 
    recommended_storage = requirements["recommended-storage"].replace("GB", "") 
    python_compatible = False
    errors = []
    warnings = []

    for python_version in profile.python_versions:
        version_str = next(iter(python_version.keys()))
        if version_str >= min_python and version_str < max_python:
            python_compatible = True

    if not python_compatible:
        errors.append(f"Python version not compatible, requires between {min_python} and {max_python}. Available versions: {', '.join([next(iter(v.keys())) for v in profile.python_versions])}")

    if profile.disk.available_gb < float(min_storage):
        errors.append(f"Insufficient disk space, at least {min_storage} GB required. Available: {profile.disk.available_gb} GB")

    if profile.memory.total_gb < float(min_ram):
        errors.append(f"Low available memory, at least {min_ram} GB recommended. Available: {profile.memory.available_gb} GB")

    if profile.disk.available_gb < float(recommended_storage):
        warnings.append(f"Low available disk space, at least {recommended_storage} GB recommended. Available: {profile.disk.available_gb} GB")

    if profile.memory.total_gb < float(recommended_ram):
        warnings.append(f"Low available memory, at least {recommended_ram} GB recommended. Available: {profile.memory.available_gb} GB")

    return EnvironmentReport(
        supported=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )

if __name__ == "__main__":
    import pathlib
    import sys

    PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    
    if not pyproject_path.exists():
        print("pyproject.toml not found")
        sys.exit(1)

    with pyproject_path.open("rb") as f:
        tomlconfig = tomllib.load(f)

    from personal_assistant.system.detector import detect_system_profile
    profile = detect_system_profile()

    report = analyze_environment(profile, tomlconfig)
    if not report.supported:
        print("Environment not supported:")
        for error in report.errors:
            print(f"  - {error}")
        for warning in report.warnings:
            print(f"  - {warning}")
        sys.exit(1)
    else:
        print("Environment is supported")
        for warning in report.warnings:
            print(f"  - {warning}")