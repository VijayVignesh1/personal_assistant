from personal_assistant.system.cpu import get_cpu_info


def test_detect_cpus():
    cpu = get_cpu_info()

    assert cpu is not None
    
    assert isinstance(cpu.model, str)
    assert hasattr(cpu, "physical_cores")
    assert isinstance(cpu.physical_cores, int)
    assert hasattr(cpu, "architecture")
    assert isinstance(cpu.architecture, str)
    assert hasattr(cpu, "logical_cores")
    assert isinstance(cpu.logical_cores, int)
    assert hasattr(cpu, "frequency_mhz")
    assert isinstance(cpu.frequency_mhz, float)
    
