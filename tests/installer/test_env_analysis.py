from personal_assistant.installer.env_analysis import analyze_environment
from personal_assistant.system.detector import detect_system_profile


def test_analyze_environment():
    profile = detect_system_profile() 
    
    report = analyze_environment(profile)
    assert report is not None
    assert hasattr(report, "supported")
    assert hasattr(report, "errors")
    assert hasattr(report, "warnings")