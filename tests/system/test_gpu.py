import pytest

from personal_assistant.system.gpu import get_gpus


@pytest.mark.gpu
def test_detect_gpus():
    gpus = get_gpus()

    for gpu in gpus:
        assert isinstance(gpu.name, str)
        assert hasattr(gpu, "memory_total_gb")
        assert isinstance(gpu.memory_total_gb, float)
        assert hasattr(gpu, "memory_available_gb")
        assert isinstance(gpu.memory_available_gb, float)
        assert hasattr(gpu, "driver_version")
        assert isinstance(gpu.driver_version, str)
        assert isinstance(gpu.is_cuda, bool)
        assert hasattr(gpu, "cuda_version")
        assert isinstance(gpu.cuda_version, str) or gpu.cuda_version is None

    assert gpus is not None
    assert isinstance(gpus, tuple)