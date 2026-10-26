import os
import cv2
import pytest
from unittest.mock import MagicMock
from src.bot_controller import BotController

USER_UPLOADED_DIR = r"C:\Users\nguye\.gemini\antigravity-ide\brain\4317e29c-fcf9-4ce3-afc3-3e8c9f2abaec\.user_uploaded"

@pytest.fixture
def dummy_bot():
    bot = BotController.__new__(BotController)
    loader = MagicMock()
    loader.load_ui_coords.return_value = {
        "combat": {
            "horse_button": {"x": 1509, "y": 410},
            "auto_button": {"x": 1306, "y": 240},
            "attack_button": {"x": 1450, "y": 770},
        }
    }
    bot._loader = loader
    from src.text_recognition import GameTextRecognizer
    bot._text = GameTextRecognizer()
    return bot

def test_horse_mounted_detection_on_foot(dummy_bot):
    assert dummy_bot._is_character_mounted_on_horse(None) is False


def test_horse_mounted_detection_with_horse_frame(dummy_bot):
    import numpy as np
    # Frame with horse color in center ROI
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    # Put horse color (BGR ~ [40, 50, 150] -> HSV Hue ~ 5, Sat ~ 170, Val ~ 150)
    frame[400:500, 750:850] = (40, 50, 150)
    assert dummy_bot._is_character_mounted_on_horse(frame) is True


def test_ensure_dismounted_from_horse_never_clicks_button(dummy_bot):
    import numpy as np
    dummy_bot._screen = MagicMock()
    # First frame has horse, second frame is dismounted
    horse_frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    horse_frame[400:500, 750:850] = (40, 50, 150)
    foot_frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    dummy_bot._screen.capture.side_effect = [horse_frame, foot_frame]

    dummy_bot._input = MagicMock()
    res = dummy_bot._ensure_dismounted_from_horse(None)
    assert res is True
    dummy_bot._input.tap.assert_not_called()
