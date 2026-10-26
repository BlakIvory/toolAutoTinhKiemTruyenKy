from unittest.mock import MagicMock, patch
import numpy as np
from src.bot_controller import BotController
from src.state_machine import BotState, StateContext
from src.text_recognition import TextBox


def test_is_inventory_full_or_overweight():
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    # Test full / overweight text
    bot._text.read.return_value = [TextBox("Hành trang của bạn quá nặng", 0.95, 500, 500)]
    assert bot._is_inventory_full_or_overweight() is True

    # Test normal text
    bot._text.read.return_value = [TextBox("Mua thành công", 0.95, 500, 500)]
    assert bot._is_inventory_full_or_overweight() is False


def test_sort_inventory_taps_sort_and_confirm():
    bot = BotController.__new__(BotController)
    bot._input = MagicMock()
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    bot._text = MagicMock()
    
    sort_box = TextBox("Sắp xếp", 0.95, 1485, 260)
    confirm_box = TextBox("Đồng ý", 0.95, 635, 600)
    bot._text.read.side_effect = [
        [sort_box],
        [confirm_box]
    ]

    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "shop": {
            "sort_button": {"x": 1485, "y": 260},
            "sort_confirm_button": {"x": 635, "y": 600}
        }
    }

    ctx = StateContext(account_id="ACC_01")
    res = bot._sort_inventory(ctx)
    assert res is True

    # Check tap "Sắp xếp" then tap "Đồng ý"
    bot._input.tap.assert_any_call(1485, 260)
    bot._input.tap.assert_any_call(635, 600)


def test_handle_buy_item_logs_warning_note_on_inventory_full():
    bot = BotController.__new__(BotController)
    bot._app_cfg = MagicMock()
    bot._app_cfg.mode.dry_run = False
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._input = MagicMock()
    bot._dismiss_unexpected_popups = MagicMock(return_value=True)
    bot._ensure_shop_and_inventory_closed = MagicMock(return_value=True)
    bot._save_progress = MagicMock()

    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "shop": {
            "dialog_buy_option": {"x": 272, "y": 566},
            "western_merchant_items": {
                "Linh Dược Tây Vực": {"x": 397, "y": 190, "row": 1, "col": 2}
            },
            "grid_slots": [{"x": 397, "y": 190}],
            "buy_button": {"x": 538, "y": 485},
            "confirm_button": {"x": 465, "y": 786},
            "close_button": {"x": 702, "y": 128},
            "inventory_close": {"x": 1577, "y": 81},
            "tooltip_title_region": {"x1": 700, "y1": 30, "x2": 1100, "y2": 150}
        }
    }

    item_tooltip = TextBox("Linh dược Tây Vực", 0.95, 850, 60)
    dong_y_mua = TextBox("Đồng ý mua", 0.95, 272, 566)
    shop_header = TextBox("Cửa hiệu Tạp hóa", 0.95, 400, 100)
    buy_btn = TextBox("Mua", 0.95, 538, 485)
    confirm_btn = TextBox("Đồng ý", 0.95, 465, 786)
    full_banner = TextBox("Hành trang của bạn quá nặng", 0.95, 500, 500)

    bot._is_inventory_full_or_overweight = MagicMock(return_value=True)

    def mock_read(frame, roi=None):
        if roi == (700, 30, 1100, 150):
            return [item_tooltip]
        if roi == (250, 520, 1350, 900):
            return [TextBox("Số lượng mua", 0.95, 500, 600), confirm_btn]
        return [dong_y_mua, shop_header, buy_btn, confirm_btn, full_banner]

    bot._text = MagicMock()
    bot._text.read.side_effect = mock_read
    bot._text.best_match.return_value = None

    ctx = StateContext(account_id="ACC_01")
    ctx.extra["requested_shop_item"] = "Linh dược Tây Vực"

    next_state = bot._handle_buy_item(ctx)

    assert next_state == BotState.RETURN_QUEST
    bot._save_progress.assert_called_with(ctx, status_text="Hành trang đầy / quá nặng ⚠️")
