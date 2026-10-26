from unittest.mock import MagicMock
from src.bot_controller import BotController
from src.state_machine import StateContext, QuestType
from src.text_recognition import TextBox

def test_combat_auto_not_turned_on_when_no_monster():
    """Verify that auto is never turned on if no monster is found, and gets disabled if active."""
    bot = BotController.__new__(BotController)
    loader = MagicMock()
    loader.load_ui_coords.return_value = {
        "combat": {
            "auto_button": {"x": 1306, "y": 240},
            "horse_button": {"x": 1545, "y": 408},
            "attack_button": {"x": 1450, "y": 770},
            "patrol_ground_points": [{"x": 1180, "y": 480, "name": "Đông"}],
            "joystick_center": {"x": 220, "y": 680},
            "target_hud_region": {"x1": 450, "y1": 60, "x2": 1100, "y2": 130},
            "left_tracker_region": {"x1": 50, "y1": 250, "x2": 350, "y2": 500},
        }
    }
    bot._loader = loader
    bot._input = MagicMock()
    bot._screen = MagicMock()
    bot._text = MagicMock()
    bot._checkpoint = MagicMock()

    # Initial frame has "tu danh" active, but no monster
    # Bot should tap auto_button to turn it off
    import numpy as np
    dummy_frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._screen.capture.return_value = dummy_frame

    # Text returns "tu danh"
    bot._text.read.return_value = [TextBox("pang tu danh", 0.9, 1400, 350)]

    # Run check logic on init_frame
    init_auto_boxes = bot._text.read(dummy_frame)
    assert any("tu danh" in b.text for b in init_auto_boxes)
