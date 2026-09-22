from dataclasses import dataclass

@dataclass
class Episode:
    episode_id: str
    started_at: str
    ended_at: str = None