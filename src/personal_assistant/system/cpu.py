import cpuinfo  # type: ignore[import-untyped]
import psutil

from personal_assistant.system.schema import CPUInfo


def get_cpu_info() -> CPUInfo:
    cpu_name = cpuinfo.get_cpu_info().get("brand_raw", "Unknown CPU")
    cpu_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else None

    return CPUInfo(
        model=cpu_name,
        architecture=cpuinfo.get_cpu_info().get("arch", "Unknown"),
        physical_cores=cpu_cores,
        logical_cores=logical_cores,
        frequency_mhz=cpu_freq
    )
    
def main() -> None:
    cpu_info = get_cpu_info()
    print(cpu_info)


if __name__ == "__main__":
    main()