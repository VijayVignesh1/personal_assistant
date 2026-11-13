import platform

from personal_assistant.system.schema import OperatingSystem


def detect_os() -> OperatingSystem:
    os_name = platform.system()
    return OperatingSystem[os_name.upper()] if os_name.upper() in OperatingSystem.__members__ else OperatingSystem.UNKNOWN

def detect_os_version() -> str:
    return platform.version()

def detect_os_architecture() -> str:
    return platform.architecture()[0]

def main() -> None:
    os = detect_os()
    os_ver = detect_os_version()
    os_arch = detect_os_architecture()
    print(os)
    print(os_ver)
    print(os_arch)


if __name__ == "__main__":
    main()