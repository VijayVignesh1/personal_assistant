from dataclasses import dataclass


@dataclass
class Message:
    content: str
    timestamp: str
    role: str
    episode_id: str
    message_id: str