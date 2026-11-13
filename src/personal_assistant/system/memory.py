import psutil

from personal_assistant.system.schema import DiskInfo, MemoryInfo


def get_memory_info() -> MemoryInfo:
    mem = psutil.virtual_memory()
    return MemoryInfo(total_gb=round(mem.total / (1024**3), 2), available_gb=round(mem.available / (1024**3), 2))

def get_disk_info() -> DiskInfo:
    disk = psutil.disk_usage('/')
    return DiskInfo(total_gb=round(disk.total / (1024**3), 2), available_gb=round(disk.free / (1024**3), 2))

def main() -> None:
    memory_info = get_memory_info()
    disk_info = get_disk_info()
    print(memory_info)
    print(disk_info)

if __name__ == "__main__":
    main()