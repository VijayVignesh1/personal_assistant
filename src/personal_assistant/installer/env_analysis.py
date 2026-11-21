
from personal_assistant.installer.schema import EnvironmentReport
from personal_assistant.system.schema import SystemProfile

MINIMUM_RAM = "8 GB"
RECOMMENDED_RAM = "16 GB"

MINIMUM_STORAGE = "2 GB"
RECOMMENDED_STORAGE = "10 GB"


def analyze_environment(
    profile: SystemProfile,
) -> EnvironmentReport:

    min_ram = MINIMUM_RAM.replace(" GB", "")
    recommended_ram = RECOMMENDED_RAM.replace(" GB", "")
    min_storage = MINIMUM_STORAGE.replace(" GB", "")
    recommended_storage = RECOMMENDED_STORAGE.replace(" GB", "")
    errors = []
    warnings = []

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
    import sys

    from personal_assistant.system.detector import detect_system_profile
    profile = detect_system_profile()

    report = analyze_environment(profile)
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