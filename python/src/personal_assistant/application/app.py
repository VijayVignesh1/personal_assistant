import datetime
import uuid
from pathlib import Path

from personal_assistant.db.db import Database
from personal_assistant.models.smolLM2 import SmolLM2
from personal_assistant.objects.episode import Episode
from personal_assistant.objects.message import Message
from personal_assistant.response.response_generator import ResponseGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DB_PATH = PROJECT_ROOT / "database" / "personal_assistant.db"

class Application:
    def __init__(self, 
                 db_path: str | Path = DB_PATH
                 ) -> None:
        """Initialize the Application class."""
        self.llm_model = SmolLM2()
        self.response_generator = ResponseGenerator(model=self.llm_model)
        self.db = Database(db_path=db_path)
        self.episode_id = str(uuid.uuid4())
        self.started_at = str(datetime.datetime.now(tz=datetime.UTC))
        self.ended_at: str | None = None

        self.db.create_episode(Episode(episode_id=self.episode_id, started_at=self.started_at))

    def generate_response(self, query: str) -> str:
        """Generate a response based on the given query."""

        # step 1: add message to the episode
        message_id = str(uuid.uuid4())
        message_time = str(datetime.datetime.now(tz=datetime.UTC))
        self.db.add_message(Message(message_id=message_id, 
                                               episode_id=self.episode_id, 
                                               content=query, 
                                               timestamp=message_time,
                                               role='user'))


        # step 2: retrieve last 10 messages for response generation
        last_messages = self.db.get_messages_by_episode(episode_id=self.episode_id)[-10:]
        context = []
        for message in last_messages:
            context.append({'role': message.role, 'content': message.content})
        
        response =  self.response_generator.generate_response(context)

        # step 3: add the generated response to the episode
        response_id = str(uuid.uuid4())
        response_time = str(datetime.datetime.now(tz=datetime.UTC))
        self.db.add_message(Message(message_id=response_id, 
                                    episode_id=self.episode_id, 
                                    content=response, 
                                    timestamp=response_time,
                                    role='assistant'))

        return response

    def close_episode(self) -> None:
        """Ensure the episode is ended when closing the episode."""
        print(f"Closing episode {self.episode_id} at {datetime.datetime.now(tz=datetime.UTC)!s}")
        self.ended_at = str(datetime.datetime.now(tz=datetime.UTC))
        self.db.close_episode(self.episode_id)