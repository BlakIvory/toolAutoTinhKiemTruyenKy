import pytest
import time
from unittest.mock import MagicMock, patch
import numpy as np

from src.bot_controller import BotController
from src.state_machine import StateContext

def test_find_western_merchant_stationary_30s():
    """Verify that in _find_western_merchant, if character is stationary for >=30s, it breaks walk and skips to next spawn."""
    bot = BotController.__new__(BotController)
    bot._input = MagicMock()
    bot._ocr = MagicMock()
    bot._dismiss_unexpected_popups = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "western_merchant_spawns": [
                {"city": "Tuyền Châu", "x": 187, "y": 191}
            ]
        }
    }
    bot._go_to_city_coordinate = MagicMock(return_value=True)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    bot._text = MagicMock()
    bot._text.read.return_value = []
    bot._text.best_match.return_value = None
    bot._checkpoint = MagicMock()
    bot._is_map_open = MagicMock(return_value=False)
    
    ctx = StateContext(account_id="ACC_01")
    ctx.current_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Stationary coordinates (100, 200) throughout
    bot._get_current_location_from_hud = MagicMock(return_value=("Tuyền Châu", (100, 200)))
    bot._is_western_merchant_shop_open = MagicMock(return_value=False)
    bot._detect_and_click_western_merchant = MagicMock(return_value=False)
    bot._ensure_western_merchant_interact_tab = MagicMock(return_value=False)
    
    # Advance time: start at 100.0, then each call advances by 5s
    cur_time = [100.0]
    def mock_time():
        t = cur_time[0]
        cur_time[0] += 5.0
        return t

    with patch("time.time", side_effect=mock_time), patch("time.sleep", return_value=None):
        res = bot._find_western_merchant(ctx)
        # Should detect stationary at 30s, break out of spawn walk, and finish with False
        assert res is False

def test_navigate_to_combat_map_stationary_30s_does_not_nest_recovery():
    """A stalled recovery must not recursively start another chief route."""
    bot = BotController.__new__(BotController)
    bot._input = MagicMock()
    bot._ocr = MagicMock()
    bot._dismiss_unexpected_popups = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "major_cities": [{"name": "Tuyền Châu"}]
        }
    }
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    bot._text = MagicMock()
    bot._text.read.return_value = []
    
    ctx = StateContext(account_id="ACC_01")
    ctx.current_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    bot._is_same_combat_map = MagicMock(return_value=False)
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    bot._is_near_commerce_chief = MagicMock(return_value=False)
    bot._get_current_location_from_hud = MagicMock(return_value=("Tuyền Châu", (150, 150)))
    bot._return_to_nearest_city_commerce_chief = MagicMock()
    bot._open_commerce_chief_dialog = MagicMock(return_value=False)
    
    # Mock time so loop runs and passes 30s
    cur_time = [0.0]
    def mock_time():
        t = cur_time[0]
        cur_time[0] += 5.0
        return t
        
    with patch("time.time", side_effect=mock_time), patch("time.sleep", return_value=None):
        bot._navigate_to_combat_map(ctx, "Kiếm Các Băng Phong")
            
        # Start one route only; nested retries caused an endless return loop.
        assert bot._return_to_nearest_city_commerce_chief.call_count == 1
