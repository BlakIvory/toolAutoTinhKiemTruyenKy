import pytest
from unittest.mock import MagicMock, patch
from src.bot_controller import BotController
from src.state_machine import StateContext

@pytest.fixture
def mock_controller():
    bot = BotController.__new__(BotController)
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "western_merchant_spawns": [
                {"city": "Biện Kinh", "x": 180, "y": 185},
                {"city": "Tuyền Châu", "x": 187, "y": 191},
            ]
        }
    }
    bot._input = MagicMock()
    bot._vision = MagicMock()
    bot._text = MagicMock()
    bot._screen = MagicMock()
    bot._checkpoint = MagicMock()
    bot._adb = MagicMock()
    bot._db = None
    bot._check_quest_timeout = MagicMock(return_value=False)
    bot._save_progress = MagicMock()
    return bot

def test_sync_quest_progress_regex(mock_controller):
    ctx = StateContext(account_id="ACC_TEST")
    ctx.total_quests_done = 5

    # Case 1: "Thương hội (13/20)"
    res = mock_controller._sync_quest_progress_from_text(ctx, "Nhiệm vụ Thương hội (13/20)", source="test")
    assert res is True
    assert ctx.total_quests_done == 13
    mock_controller._save_progress.assert_called_with(ctx, status_text="Đang làm NV #14/20")

    # Case 2: "Còn có thể nhận 5 lần!" -> 15/20
    res = mock_controller._sync_quest_progress_from_text(ctx, "Hôm nay còn có thể nhận 5 lần!", source="test")
    assert res is True
    assert ctx.total_quests_done == 15
    mock_controller._save_progress.assert_called_with(ctx, status_text="Đang làm NV #16/20")

    # Case 3: OCR 'O' instead of '0' -> "16/2O"
    res = mock_controller._sync_quest_progress_from_text(ctx, "Thương hội 16/2O", source="test")
    assert res is True
    assert ctx.total_quests_done == 16
    mock_controller._save_progress.assert_called_with(ctx, status_text="Đang làm NV #17/20")

def test_find_western_merchant_fast_exit(mock_controller):
    ctx = StateContext(account_id="ACC_TEST")
    ctx.extra["quest_text"] = "Mua đồ Lông Đại Bàng ở Biện Kinh"

    mock_controller._get_current_location_from_hud = MagicMock(return_value=("Biện Kinh", (180, 184)))
    mock_controller._detect_current_city = MagicMock(return_value="Biện Kinh")
    mock_controller._go_to_city_coordinate = MagicMock(return_value=True)
    mock_controller._detect_and_interact_merchant = MagicMock(return_value=False)
    # Ghost bubble present at (1220, 299)
    mock_controller._is_npc_interact_bubble_present = MagicMock(return_value=True)
    mock_controller._click_npc_interact_bubble = MagicMock(return_value=True)
    # But clicking bubble does NOT open merchant dialog
    mock_controller._is_western_merchant_dialog = MagicMock(return_value=False)
    mock_controller._is_npc_dialog_open = MagicMock(return_value=False)
    mock_controller._ensure_npc_dialog_closed = MagicMock()

    # Run _find_western_merchant
    with patch("time.sleep", return_value=None):
        result = mock_controller._find_western_merchant(ctx)

    # Should not find merchant and exit safely
    assert result is False
    # Bubble should only be clicked at most ONCE per city (not 4 times!)
    # Since there are 2 spawns (Biện Kinh and Tuyền Châu), total clicks should be <= 2
    assert mock_controller._click_npc_interact_bubble.call_count <= 2
