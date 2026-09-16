from unittest.mock import MagicMock, patch
from src.bot_controller import BotController
from src.state_machine import StateContext
import numpy as np


def test_switch_to_city_skips_if_already_there():
    bot = BotController.__new__(BotController)
    bot._get_current_location_from_hud = MagicMock(return_value=("Tuyền Châu", (192, 182)))
    bot._ensure_map_open = MagicMock()
    bot._input = MagicMock()

    ctx = StateContext(account_id="ACC_01")
    res = bot._switch_to_city(ctx, "Tuyền Châu")
    assert res is True
    bot._ensure_map_open.assert_not_called()


def test_switch_to_city_navigates_when_different():
    bot = BotController.__new__(BotController)
    bot._get_current_location_from_hud = MagicMock(return_value=("Tuyền Châu", (192, 182)))
    bot._dismiss_unexpected_popups = MagicMock()
    bot._ensure_map_open = MagicMock()
    bot._input = MagicMock()
    bot._adb = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "navigation": {
            "map_tab_find_map": {"x": 176, "y": 162},
            "map_search_input": {"x": 266, "y": 222},
        }
    }
    bot._input_search_text = MagicMock()
    bot._click_map_search_result = MagicMock(return_value=True)

    ctx = StateContext(account_id="ACC_01")
    res = bot._switch_to_city(ctx, "Biện Kinh")
    assert res is True
    bot._ensure_map_open.assert_called_once()
    bot._input_search_text.assert_called_once()
    bot._click_map_search_result.assert_called_once_with("Biện Kinh")


def test_go_to_city_coordinate_inputs_exact_coords():
    bot = BotController.__new__(BotController)
    bot._dismiss_unexpected_popups = MagicMock()
    bot._switch_to_city = MagicMock(return_value=True)
    bot._ensure_map_open = MagicMock()
    bot._is_map_open = MagicMock(return_value=False)
    bot._close_map = MagicMock()
    bot._press_buff_speed = MagicMock()
    bot._input = MagicMock()
    bot._adb = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "map_coordinate_x": {"x": 704, "y": 778},
            "map_coordinate_y": {"x": 842, "y": 778},
            "map_coordinate_confirm": {"x": 1015, "y": 778},
        }
    }

    ctx = StateContext(account_id="ACC_01")
    res = bot._go_to_city_coordinate(ctx, "Tuyền Châu", 187, 191)
    assert res is True
    bot._switch_to_city.assert_called_once_with(ctx, "Tuyền Châu", force=True)
    # Verify taps X and Y input boxes and confirm
    bot._input.tap.assert_any_call(704, 778)
    bot._input.tap.assert_any_call(842, 778)
    bot._input.tap.assert_any_call(1015, 778)
    bot._input.input_text.assert_any_call("187")
    bot._input.input_text.assert_any_call("191")
    bot._press_buff_speed.assert_called_once()


def test_find_western_merchant_walks_and_finds():
    bot = BotController.__new__(BotController)
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
    from src.text_recognition import TextBox
    # Found merchant immediately
    merchant_box = TextBox("Thương nhân Tây Vực", 0.95, 750, 420)
    bot._text.read.return_value = [merchant_box]
    bot._text.best_match.return_value = (0.95, merchant_box, "Thương nhân Tây Vực")
    bot._get_current_location_from_hud = MagicMock(return_value=("Tuyền Châu", (187, 191)))
    bot._input = MagicMock()
    bot._checkpoint = MagicMock()

    ctx = StateContext(account_id="ACC_01")
    res = bot._find_western_merchant(ctx)
    assert res is True
    assert ctx.extra.get("merchant_found_city") == "Tuyền Châu"
    bot._input.tap.assert_called_once_with(750, 455)


def test_handle_buy_item_clicks_dong_y_mua_and_confirms():
    from src.state_machine import BotState
    from src.text_recognition import TextBox

    bot = BotController.__new__(BotController)
    bot._app_cfg = MagicMock()
    bot._app_cfg.mode.dry_run = False
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "shop": {
            "dialog_buy_option": {"x": 272, "y": 566},
            "item_slot_linh_duoc": {"x": 397, "y": 190},
            "buy_button": {"x": 538, "y": 485},
            "confirm_button": {"x": 465, "y": 786},
            "close_button": {"x": 702, "y": 128},
            "inventory_close": {"x": 1577, "y": 81},
        }
    }
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    bot._text = MagicMock()
    bot._text.read.return_value = []
    bot._text.best_match.return_value = None
    bot._vision = MagicMock()
    bot._vision.template_exists.return_value = False
    bot._input = MagicMock()
    bot._adb = MagicMock()
    bot._dismiss_unexpected_popups = MagicMock()

    ctx = StateContext(account_id="ACC_01")
    ctx.extra["requested_shop_item"] = "Linh dược Tây Vực"

    next_state = bot._handle_buy_item(ctx)

    assert next_state == BotState.RETURN_QUEST
    assert ctx.extra.get("returning_quest") is True
    # Verify exact taps:
    # 1. Dialog 'Đồng ý mua' (272, 566)
    # 2. Item slot 2 (397, 190)
    # 3. Buy button (538, 485)
    # 4. Confirm 'Đồng ý' (465, 786)
    # 5. Close X (702, 128)
    bot._input.tap.assert_any_call(272, 566)
    bot._input.tap.assert_any_call(397, 190)
    bot._input.tap.assert_any_call(538, 485)
    bot._input.tap.assert_any_call(465, 786)
    bot._input.tap.assert_any_call(702, 128)


def test_handle_buy_item_inspects_slots_until_match():
    from src.state_machine import BotState
    from src.text_recognition import TextBox

    bot = BotController.__new__(BotController)
    bot._app_cfg = MagicMock()
    bot._app_cfg.mode.dry_run = False
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "shop": {
            "dialog_buy_option": {"x": 272, "y": 566},
            "buy_button": {"x": 538, "y": 485},
            "confirm_button": {"x": 465, "y": 786},
            "close_button": {"x": 702, "y": 128},
            "grid_slots": [
                {"x": 316, "y": 190},  # Slot 1
                {"x": 397, "y": 190},  # Slot 2 (Target)
                {"x": 486, "y": 190},  # Slot 3
            ],
            "tooltip_title_region": {"x1": 700, "y1": 30, "x2": 1100, "y2": 150},
        }
    }
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    # When slot 1 is clicked -> tooltip shows "Đoản đao"
    # When slot 2 is clicked -> tooltip shows "Linh dược Tây Vực"
    def mock_read(frame, roi=None):
        if roi == (700, 30, 1100, 150):
            # Check which slot was tapped last:
            last_tap = bot._input.tap.call_args_list[-1][0]
            if last_tap == (316, 190):
                return [TextBox("Đoản đao", 0.9, 850, 60)]
            elif last_tap == (397, 190):
                return [TextBox("Linh dược Tây Vực", 0.95, 850, 60)]
        return []

    bot._text = MagicMock()
    bot._text.read.side_effect = mock_read
    bot._text.best_match.return_value = None
    bot._vision = MagicMock()
    bot._vision.template_exists.return_value = False
    bot._input = MagicMock()
    bot._adb = MagicMock()
    bot._dismiss_unexpected_popups = MagicMock()

    ctx = StateContext(account_id="ACC_01")
    ctx.extra["requested_shop_item"] = "Linh dược Tây Vực"

    next_state = bot._handle_buy_item(ctx)

    assert next_state == BotState.RETURN_QUEST
    # Assert slot 1 was checked first
    bot._input.tap.assert_any_call(316, 190)
    # Assert slot 2 was checked second and matched
    bot._input.tap.assert_any_call(397, 190)
    # Assert slot 3 was NEVER checked because slot 2 already matched!
    tapped_coords = [call[0] for call in bot._input.tap.call_args_list]
    assert (486, 190) not in tapped_coords
    # Assert "Mua" was clicked
    bot._input.tap.assert_any_call(538, 485)
    # Assert "Đồng ý" was clicked
    bot._input.tap.assert_any_call(465, 786)
