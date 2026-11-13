from dataclasses import dataclass
from enum import Enum


class OperatingSystem(str, Enum):
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    UNKNOWN = "unknown"


class GPUVendor(str, Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    UNKNOWN = "unknown"

@dataclass
class CPUInfo:
    architecture: str
    logical_cores: int | None
    model: str | None = None
    physical_cores: int | None = None
    frequency_mhz: float | None = None
    

@dataclass
class MemoryInfo:
    total_gb: float
    available_gb: float


@dataclass
class GPUInfo:
    name: str
    vendor: GPUVendor
    memory_total_gb: float | None = None
    memory_available_gb: float | None = None
    driver_version: str | None = None
    is_cuda: bool | None = False
    cuda_version: str | None = None


@dataclass
class DiskInfo:
    total_gb: float
    available_gb: float


@dataclass
class SystemProfile:
    os: OperatingSystem
    os_version: str
    os_architecture: str

    cpu: CPUInfo
    memory: MemoryInfo
    disk: DiskInfo

    python_version: str

    gpus: tuple[GPUInfo,...] = ()