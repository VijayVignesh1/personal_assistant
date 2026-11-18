from personal_assistant.installer.env_analysis import analyze_environment
from personal_assistant.system.detector import detect_system_profile


def test_analyze_environment():
    profile = detect_system_profile()
    tomlconfig = {
        "project": {"name": "personal_assistant",
                    "requires-python": ">=3.11,<3.14"},
        "tool": {"personal-assistant": {"requirements": {"minimum-ram": "8 GB",
                                                         "recommended-ram": "16 GB",
                                                         "minimum-storage": "2 GB",
                                                         "recommended-storage": "10 GB"}}}
    }  
    
    report = analyze_environment(profile, tomlconfig)
    assert report is not None
    assert hasattr(report, "supported")
    assert hasattr(report, "errors")
    assert hasattr(report, "warnings")