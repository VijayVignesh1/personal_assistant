from personal_assistant.system.memory import get_disk_info, get_memory_info


def test_get_memory_info():
    memory = get_memory_info()

    assert memory is not None
    assert hasattr(memory, "total_gb")
    assert isinstance(memory.total_gb, float)
    assert hasattr(memory, "available_gb")
    assert isinstance(memory.available_gb, float)


def test_get_disk_info():
    disk = get_disk_info()

    assert disk is not None
    assert hasattr(disk, "total_gb")
    assert isinstance(disk.total_gb, float)
    assert hasattr(disk, "available_gb")
    assert isinstance(disk.available_gb, float)