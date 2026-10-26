from src.text_recognition import GameTextRecognizer, TextBox


def test_tuyen_chau_map_name_tolerates_missing_accent():
    boxes = [
        TextBox("Dao Hoa dao", .95, 170, 390),
        TextBox("Tuyen Chau", .90, 170, 330),
    ]
    result = GameTextRecognizer.best_match(
        boxes, ["Tuyền Châu", "Tuyên Châu"], minimum=.48)
    assert result is not None
    assert result[1].center_y == 330


def test_route_can_reject_similar_submap_by_row():
    boxes = [TextBox("Tay Tuyen Chau", .95, 170, 418)]
    result = GameTextRecognizer.best_match(boxes, ["Tuyền Châu"], minimum=.48)
    assert result is not None
    assert not (300 <= result[1].center_y <= 365)


def test_navigate_to_tuyen_chau_commerce_chief_executes_8_steps():
    from unittest.mock import MagicMock
    from src.bot_controller import BotController
    from src.state_machine import StateContext

    bot = BotController.__new__(BotController)
    bot._ensure_map_open = MagicMock()
    bot._close_map = MagicMock()
    bot._press_buff_speed = MagicMock()
    bot._input = MagicMock()
    bot._adb = MagicMock()
    bot._screen = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "navigation": {
            "map_tab_find_map": {"x": 176, "y": 162},
            "map_tab_find_npc": {"x": 376, "y": 161},
            "map_search_input": {"x": 266, "y": 222},
            "map_tuyen_chau": {"x": 115, "y": 315},
            "map_npc_search_result": {"x": 200, "y": 285},
        }
    }
    bot._input_search_text = MagicMock()
    bot._click_map_search_result = MagicMock(return_value=True)
    bot._click_npc_search_result = MagicMock(return_value=True)

    ctx = StateContext(account_id="ACC_01")
    result = bot._navigate_to_tuyen_chau_commerce_chief(ctx)

    assert result is True
    # Verify Step 1: Open map
    assert bot._ensure_map_open.call_count >= 1
    # Verify Step 2 & 3: Taps find map & search input
    bot._input.tap.assert_any_call(176, 162)
    # Verify Step 4: Types Tuyen Chau with unicode accents
    bot._input_search_text.assert_any_call("Tuyen Chau", unicode_text="Tuyền Châu", ctx=ctx)
    # Verify Step 5: Clicks Tuyen Chau map
    bot._click_map_search_result.assert_called_once_with("Tuyền Châu")
    # Verify Step 6: Taps find NPC tab
    bot._input.tap.assert_any_call(376, 161)
    # Verify Step 8: Types Chu thuong hoi with unicode accents
    bot._input_search_text.assert_any_call("Chu thuong hoi", unicode_text="Chủ thương hội", ctx=ctx)
    # Verify Step 9: Clicks Chu thuong hoi NPC directly from list
    bot._click_npc_search_result.assert_called_once_with("Chu thuong hoi")
    # Verify Close map (buff speed is intentionally not clicked to avoid cancelling pathfinding)
    bot._close_map.assert_called_once()


def test_return_to_chief_redirects_to_tuyen_chau_if_in_tuong_duong():
    from unittest.mock import MagicMock
    from src.bot_controller import BotController
    from src.state_machine import StateContext

    bot = BotController.__new__(BotController)
    bot._is_near_commerce_chief = MagicMock(return_value=False)
    bot._get_current_location_from_hud = MagicMock(return_value=("tdong cuTuong", (178, 182)))
    bot._dismiss_unexpected_popups = MagicMock()
    bot._navigate_to_tuyen_chau_commerce_chief = MagicMock(return_value=True)

    ctx = StateContext(account_id="ACC_02")
    res = bot._return_to_nearest_city_commerce_chief(ctx)

    assert res is True
    bot._navigate_to_tuyen_chau_commerce_chief.assert_called_once_with(ctx)


def test_click_npc_search_result_rejects_unrelated_npc():
    from unittest.mock import MagicMock
    import numpy as np
    from src.bot_controller import BotController
    from src.text_recognition import TextBox

    bot = BotController.__new__(BotController)
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {"navigation": {}}
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._text = MagicMock()
    # Screen shows "Chủ tiệm vũ khí" at (148, 318) instead of "Chủ thương hội"
    bot._text.read.return_value = [
        TextBox("Chủ tiệm vũ khí", 0.95, 148, 318),
    ]
    bot._input = MagicMock()

    res = bot._click_npc_search_result("Chu thuong hoi")
    assert res is False
    bot._input.tap.assert_not_called()


def test_is_near_commerce_chief_rejects_tuong_duong_weapon_shop_coords():
    from unittest.mock import MagicMock
    import numpy as np
    from src.bot_controller import BotController
    from src.text_recognition import TextBox

    bot = BotController.__new__(BotController)
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    bot._vision = MagicMock()
    bot._vision.find_template.return_value = MagicMock(found=False)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._get_current_location_from_hud = MagicMock(return_value=("Tương Dương", (178, 182)))
    bot._text = MagicMock()
    # Coordinates in Tương Dương: (178, 182)
    bot._text.read.return_value = [
        TextBox("Tương Dương", 0.95, 1450, 45),
        TextBox("178/182", 0.95, 1450, 80),
    ]

    assert bot._is_near_commerce_chief() is False


def test_is_near_commerce_chief_tuyen_chau_distance_threshold():
    from unittest.mock import MagicMock
    import numpy as np
    from src.bot_controller import BotController

    bot = BotController.__new__(BotController)
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    bot._vision = MagicMock()
    bot._vision.find_template.return_value = MagicMock(found=False)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._text = MagicMock()
    bot._text.read.return_value = []
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {}

    # dist = 3: (192, 179) - tại cổng thành gần Vệ binh thành môn -> Phải từ chối (False)
    bot._get_current_location_from_hud = MagicMock(return_value=("Tuyền Châu", (192, 179)))
    assert bot._is_near_commerce_chief() is False

    # dist = 2: (192, 184) -> Phải từ chối vì dist > 1 (False)
    bot._get_current_location_from_hud = MagicMock(return_value=("Tuyền Châu", (192, 184)))
    assert bot._is_near_commerce_chief() is False

    # dist = 1: (192, 183) -> Chấp nhận (True)
    bot._get_current_location_from_hud = MagicMock(return_value=("Tuyền Châu", (192, 183)))
    assert bot._is_near_commerce_chief() is True

    # dist = 1: (191, 182) -> Chấp nhận (True)
    bot._get_current_location_from_hud = MagicMock(return_value=("Tuyền Châu", (191, 182)))
    assert bot._is_near_commerce_chief() is True

    # dist = 0: (192, 182) -> Chấp nhận (True)
    bot._get_current_location_from_hud = MagicMock(return_value=("Tuyền Châu", (192, 182)))
    assert bot._is_near_commerce_chief() is True


def test_duong_chau_chief_arrival_nudges_northeast_to_reveal_npc():
    from unittest.mock import MagicMock, patch
    from src.bot_controller import BotController
    from src.state_machine import StateContext

    bot = BotController.__new__(BotController)
    bot._get_current_location_from_hud = MagicMock(
        return_value=("Dương Châu", (204, 196))
    )
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "joystick_center": {"x": 220, "y": 680},
            "commerce_chief_arrival_nudge": {
                "Dương Châu": {
                    "x": 204, "y": 196, "tolerance": 1,
                    "dx": 70, "dy": -70, "duration_ms": 500,
                }
            },
        }
    }
    bot._input = MagicMock()
    ctx = StateContext(account_id="ACC_03")

    with patch("src.bot_controller.time.sleep"):
        assert bot._nudge_to_reveal_commerce_chief(ctx) is True

    bot._input.swipe.assert_called_once_with(220, 680, 290, 610, duration=500)
    assert ctx.extra["chief_arrival_nudge_attempts"] == 1

    with patch("src.bot_controller.time.sleep"):
        assert bot._nudge_to_reveal_commerce_chief(ctx) is True
        assert bot._nudge_to_reveal_commerce_chief(ctx) is False

    assert bot._input.swipe.call_count == 2
    assert ctx.extra["chief_arrival_nudge_attempts"] == 2


def test_is_near_commerce_chief_accepts_duong_chau_arrival_boundary():
    from unittest.mock import MagicMock
    import numpy as np
    from src.bot_controller import BotController

    bot = BotController.__new__(BotController)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    bot._text = MagicMock()
    bot._text.read.return_value = []
    bot._get_current_location_from_hud = MagicMock(
        return_value=("Dương Châu", (205, 196))
    )
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "commerce_chief_arrival_nudge": {
                "Dương Châu": {"x": 204, "y": 196, "tolerance": 1}
            }
        }
    }

    assert bot._is_near_commerce_chief() is True


def test_is_near_commerce_chief_rejects_duong_chau_outside_one_point():
    from unittest.mock import MagicMock
    import numpy as np
    from src.bot_controller import BotController

    bot = BotController.__new__(BotController)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    bot._text = MagicMock()
    bot._text.read.return_value = []
    bot._get_current_location_from_hud = MagicMock(
        return_value=("Dương Châu", (206, 196))
    )
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "commerce_chief_arrival_nudge": {
                "Dương Châu": {"x": 204, "y": 196, "tolerance": 1}
            }
        }
    }

    assert bot._is_near_commerce_chief() is False


def test_navigate_to_tuyen_chau_switches_city_if_in_tuong_duong():
    from unittest.mock import MagicMock
    from src.bot_controller import BotController
    from src.state_machine import StateContext

    bot = BotController.__new__(BotController)
    bot._get_current_location_from_hud = MagicMock(return_value=("Tương Dương", (171, 191)))
    bot._switch_to_city = MagicMock(return_value=True)
    bot._ensure_map_open = MagicMock()
    bot._close_map = MagicMock()
    bot._press_buff_speed = MagicMock()
    bot._input = MagicMock()
    bot._adb = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {"navigation": {}}
    bot._input_search_text = MagicMock()
    bot._click_map_search_result = MagicMock(return_value=True)
    bot._click_npc_search_result = MagicMock(return_value=True)

    ctx = StateContext(account_id="ACC_02")
    res = bot._navigate_to_tuyen_chau_commerce_chief(ctx)

    assert res is True
    bot._switch_to_city.assert_called_once_with(
        ctx,
        "Tuyền Châu",
        current_location=("Tương Dương", None),
    )


def test_returning_quest_skips_xa_phu_and_searches_tuyen_chau_chief_directly():
    from unittest.mock import MagicMock
    from src.bot_controller import BotController
    from src.state_machine import StateContext

    bot = BotController.__new__(BotController)
    bot._get_current_location_from_hud = MagicMock(
        return_value=("Thành Đô", (235, 215))
    )
    bot._switch_to_city = MagicMock(return_value=False)
    bot._ensure_map_open = MagicMock(return_value=True)
    bot._close_map = MagicMock()
    bot._input = MagicMock()
    bot._adb = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {"navigation": {}}
    bot._input_search_text = MagicMock()
    bot._click_map_search_result = MagicMock(return_value=True)
    bot._click_npc_search_result = MagicMock(return_value=True)

    ctx = StateContext(account_id="ACC_03")
    ctx.extra["returning_quest"] = True
    ctx.extra["out_of_money"] = True

    assert bot._navigate_to_tuyen_chau_commerce_chief(ctx) is True
    bot._switch_to_city.assert_not_called()
    bot._click_map_search_result.assert_called_once_with("Tuyền Châu")
    bot._click_npc_search_result.assert_called_once_with("Chu thuong hoi")


def test_navigate_to_chief_stops_when_minimap_cannot_open():
    from unittest.mock import MagicMock
    from src.bot_controller import BotController
    from src.state_machine import StateContext

    bot = BotController.__new__(BotController)
    bot._get_current_location_from_hud = MagicMock(return_value=("Tuyền Châu", (190, 180)))
    bot._ensure_map_open = MagicMock(return_value=False)
    bot._input = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {"navigation": {}}

    result = bot._navigate_to_tuyen_chau_commerce_chief(StateContext(account_id="ACC_02"))

    assert result is False
    bot._input.tap.assert_not_called()


def test_find_chief_does_not_mark_navigation_started_when_minimap_failed():
    from unittest.mock import MagicMock, patch
    from src.bot_controller import BotController
    from src.state_machine import BotState, StateContext

    bot = BotController.__new__(BotController)
    bot._is_near_commerce_chief = MagicMock(return_value=False)
    bot._navigate_to_tuyen_chau_commerce_chief = MagicMock(return_value=False)
    ctx = StateContext(account_id="ACC_02")
    ctx.extra["check_previous_quest_on_login"] = True

    with patch("src.bot_controller.time.sleep"):
        state = bot._handle_find_commerce_chief(ctx)

    assert state == BotState.FIND_COMMERCE_CHIEF
    assert "chief_nav_started" not in ctx.extra
    assert ctx.extra["chief_nav_failures"] == 1


def test_find_chief_recovers_after_three_consecutive_navigation_failures():
    from unittest.mock import MagicMock, patch
    from src.bot_controller import BotController
    from src.state_machine import BotState, StateContext

    bot = BotController.__new__(BotController)
    bot._is_near_commerce_chief = MagicMock(return_value=False)
    bot._navigate_to_tuyen_chau_commerce_chief = MagicMock(return_value=False)
    ctx = StateContext(account_id="ACC_03")
    ctx.extra["chief_nav_failures"] = 2

    with patch("src.bot_controller.time.sleep"):
        state = bot._handle_find_commerce_chief(ctx)

    assert state == BotState.RECOVERY
    assert "chief_nav_failures" not in ctx.extra


def test_return_to_chief_stays_in_bien_kinh_if_chief_found():
    from unittest.mock import MagicMock
    from src.bot_controller import BotController
    from src.state_machine import StateContext

    bot = BotController.__new__(BotController)
    bot._is_near_commerce_chief = MagicMock(return_value=False)
    bot._get_current_location_from_hud = MagicMock(return_value=("Biện Kinh", (183, 186)))
    bot._ensure_map_open = MagicMock(return_value=True)
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {"navigation": {}}
    bot._input = MagicMock()
    bot._adb = MagicMock()
    bot._input_search_text = MagicMock()
    bot._click_npc_search_result = MagicMock(return_value=True)
    bot._navigate_to_tuyen_chau_commerce_chief = MagicMock()

    ctx = StateContext(account_id="ACC_01")
    res = bot._return_to_nearest_city_commerce_chief(ctx)

    assert res is True
    assert ctx.extra.get("chief_city") == "Biện Kinh"
    bot._click_npc_search_result.assert_called_once_with("Chu thuong hoi")
    bot._navigate_to_tuyen_chau_commerce_chief.assert_not_called()


def test_return_to_chief_falls_back_to_tuyen_chau_if_not_in_bien_kinh():
    from unittest.mock import MagicMock
    from src.bot_controller import BotController
    from src.state_machine import StateContext

    bot = BotController.__new__(BotController)
    bot._is_near_commerce_chief = MagicMock(return_value=False)
    bot._get_current_location_from_hud = MagicMock(return_value=("Biện Kinh", (183, 186)))
    bot._ensure_map_open = MagicMock(return_value=True)
    bot._is_map_open = MagicMock(return_value=True)
    bot._close_map = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {"navigation": {}}
    bot._input = MagicMock()
    bot._adb = MagicMock()
    bot._input_search_text = MagicMock()
    bot._click_npc_search_result = MagicMock(return_value=False)
    bot._navigate_to_tuyen_chau_commerce_chief = MagicMock(return_value=True)

    ctx = StateContext(account_id="ACC_01")
    res = bot._return_to_nearest_city_commerce_chief(ctx)

    assert res is True
    assert ctx.extra.get("chief_city") == "Tuyền Châu"
    bot._navigate_to_tuyen_chau_commerce_chief.assert_called_once_with(ctx)


def test_find_chief_interacts_in_bien_kinh_when_chief_city_is_bien_kinh():
    from unittest.mock import MagicMock, patch
    from src.bot_controller import BotController
    from src.state_machine import BotState, StateContext

    bot = BotController.__new__(BotController)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = "dummy_frame"
    bot._dismiss_unexpected_popups = MagicMock()
    bot._get_current_location_from_hud = MagicMock(return_value=("Biện Kinh", (170, 182)))
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    bot._is_npc_interact_bubble_present = MagicMock(return_value=True)
    bot._open_commerce_chief_dialog = MagicMock(return_value=True)
    bot._verify_commerce_chief_dialog = MagicMock(return_value=True)
    bot._switch_to_city = MagicMock()
    bot._nudge_to_reveal_commerce_chief = MagicMock(return_value=False)

    ctx = StateContext(account_id="ACC_04")
    ctx.extra["chief_city"] = "Biện Kinh"
    ctx.extra["chief_nav_started"] = True

    with patch("src.bot_controller.time.sleep"):
        state = bot._handle_find_commerce_chief(ctx)

    assert state == BotState.MEET_NPC
    bot._switch_to_city.assert_not_called()
    bot._open_commerce_chief_dialog.assert_called_once_with(ctx)
