from dataclasses import dataclass


@dataclass
class EnvironmentReport:
    supported: bool
    errors: list[str]
    warnings: list[str]
