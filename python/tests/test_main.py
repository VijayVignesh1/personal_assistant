from unittest.mock import patch

import pytest

from personal_assistant.application.app import Application
from personal_assistant.main import main


@pytest.fixture
def application(tmp_path):
    db_path = tmp_path / "test.db"
    return  Application(db_path=str(db_path))

def test_ctrl_c_ends_episode(application):

    episode_id = application.episode_id
    db = application.db

    with patch("personal_assistant.main.Application", return_value=application), \
         patch("personal_assistant.main.UI") as mock_ui:
        mock_ui.return_value.launch.side_effect = KeyboardInterrupt

        try:
            main()
        except KeyboardInterrupt:
            pass

    episode = db.get_episode(episode_id)

    assert episode is not None
    assert episode.ended_at is not None
    assert isinstance(episode.ended_at, str)