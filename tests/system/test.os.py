from personal_assistant.system.os import detect_os, detect_os_architecture, detect_os_version


def test_detect_os():
    os_info = detect_os()
    assert os_info is not None

def test_detect_os_version():
    version = detect_os_version()
    assert version is not None
    assert isinstance(version, str)

def test_detect_os_architecture():
    arch = detect_os_architecture()
    assert arch is not None
    assert isinstance(arch, str)