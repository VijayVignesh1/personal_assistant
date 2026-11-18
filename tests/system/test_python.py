from personal_assistant.system.python import find_python_versions


def test_find_python_versions():
    versions = find_python_versions()
    assert isinstance(versions, list)
    for python in versions:
        assert isinstance(python, dict)
        assert len(python) == 1
        version_str = next(iter(python.keys()))
        assert isinstance(version_str, str)
        assert version_str.count(".") == 2  # Ensure version is in the format X.Y.Z
        assert isinstance(python[version_str], str)