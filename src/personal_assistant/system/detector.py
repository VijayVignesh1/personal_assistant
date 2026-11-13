import sys

from personal_assistant.system.cpu import get_cpu_info
from personal_assistant.system.gpu import get_gpus
from personal_assistant.system.memory import get_disk_info, get_memory_info
from personal_assistant.system.os import detect_os, detect_os_architecture, detect_os_version
from personal_assistant.system.schema import SystemProfile


def detect_system_profile() -> SystemProfile:
    os_info = detect_os()
    memory_info = get_memory_info()
    disk_info = get_disk_info()
    cpu_info = get_cpu_info()
    gpu_list = get_gpus()
    os_version = detect_os_version()
    os_architecture = detect_os_architecture()
    python_version = sys.version
    return SystemProfile(
        os=os_info,
        os_version=os_version,
        os_architecture=os_architecture,
        memory=memory_info,
        disk=disk_info,
        cpu=cpu_info,
        gpus=gpu_list,
        python_version=python_version
    )

def main() -> None:
    system_profile = detect_system_profile()
    print(system_profile)

if __name__ == "__main__":
    main()