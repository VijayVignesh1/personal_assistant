from personal_assistant.system.detector import detect_system_profile
from personal_assistant.system.schema import CPUInfo, DiskInfo, MemoryInfo


def test_detect_system_profile():
    profile = detect_system_profile()
    assert profile is not None
    assert hasattr(profile, "gpus")
    assert isinstance(profile.gpus, tuple)
    assert isinstance(profile.cpu, CPUInfo)
    assert hasattr(profile, "memory")
    assert isinstance(profile.memory, MemoryInfo)
    assert hasattr(profile, "disk")
    assert isinstance(profile.disk, DiskInfo)
    assert hasattr(profile, "os_architecture")
    assert isinstance(profile.os_architecture, str)
    assert isinstance(profile.os_version, str)
    assert isinstance(profile.python_version, str)