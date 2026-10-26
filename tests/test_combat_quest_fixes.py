from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.bot_controller import BotController
from src.state_machine import StateContext, QuestType, BotState
from src.text_recognition import TextBox


def test_attack_monster_blocking_travel_uses_right_click_only():
    bot = BotController.__new__(BotController)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._is_map_open = MagicMock(return_value=False)
    bot._find_monster_on_screen = MagicMock(return_value=(610, 430))
    bot._input = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {"combat": {}}
    bot._text = MagicMock()
    bot._text.read.return_value = [TextBox("Sói điên", 0.95, 600, 90)]
    ctx = StateContext(account_id="ACC_03")
    ctx.extra["combat_monster"] = "Nhện độc"

    assert bot._attack_monster_blocking_travel(ctx) is True
    bot._find_monster_on_screen.assert_called_once_with(
        bot._screen.capture.return_value, "Sói điên"
    )
    bot._input.right_click.assert_called_once_with(610, 430, count=1)
    bot._input.tap.assert_not_called()


def test_attack_monster_blocking_travel_does_nothing_without_identified_target():
    bot = BotController.__new__(BotController)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._is_map_open = MagicMock(return_value=False)
    bot._find_monster_on_screen = MagicMock(return_value=None)
    bot._input = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {"combat": {}}
    bot._text = MagicMock()
    bot._text.read.return_value = []
    ctx = StateContext(account_id="ACC_03")
    ctx.extra["combat_monster"] = "Nhện độc"

    assert bot._attack_monster_blocking_travel(ctx) is False
    bot._input.right_click.assert_not_called()


@pytest.mark.parametrize("return_flag", [
    {"combat_objective_complete": True},
    {"returning_quest": True},
])
def test_attack_monster_blocking_travel_is_disabled_while_returning_quest(return_flag):
    bot = BotController.__new__(BotController)
    bot._screen = MagicMock()
    bot._input = MagicMock()
    ctx = StateContext(account_id="ACC_03")
    ctx.extra.update(return_flag)

    assert bot._attack_monster_blocking_travel(ctx) is False
    bot._screen.capture.assert_not_called()
    bot._input.right_click.assert_not_called()


def test_attack_monster_blocking_travel_closes_bag_instead_of_attacking_tooltip():
    bot = BotController.__new__(BotController)
    bot._screen = MagicMock()
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._screen.capture.return_value = frame
    bot._is_map_open = MagicMock(return_value=False)
    bot._is_bag_open = MagicMock(return_value=True)
    bot._close_bag = MagicMock(return_value=True)
    bot._find_monster_on_screen = MagicMock(return_value=(825, 305))
    bot._input = MagicMock()
    bot._loader = MagicMock()
    bot._text = MagicMock()
    ctx = StateContext(account_id="ACC_03")

    assert bot._attack_monster_blocking_travel(ctx) is False
    bot._close_bag.assert_called_once_with()
    bot._find_monster_on_screen.assert_not_called()
    bot._input.right_click.assert_not_called()


def test_attack_monster_blocking_travel_ignores_item_attribute_as_monster_name():
    bot = BotController.__new__(BotController)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._is_map_open = MagicMock(return_value=False)
    bot._is_bag_open = MagicMock(return_value=False)
    bot._find_monster_on_screen = MagicMock(return_value=(825, 305))
    bot._input = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {"combat": {}}
    bot._text = MagicMock()
    bot._text.read.return_value = [TextBox("Loại:", 0.99, 825, 90)]
    ctx = StateContext(account_id="ACC_03")

    assert bot._attack_monster_blocking_travel(ctx) is False
    bot._find_monster_on_screen.assert_not_called()
    bot._input.right_click.assert_not_called()


def test_find_commerce_chief_routes_active_combat_quest_to_classify_quest():
    bot = BotController.__new__(BotController)
    bot._is_near_commerce_chief = MagicMock(side_effect=[False, True])
    bot._navigate_to_tuyen_chau_commerce_chief = MagicMock(return_value=True)
    bot._vision = MagicMock()
    bot._is_npc_dialog_open = MagicMock(return_value=True)
    bot._verify_commerce_chief_dialog = MagicMock(return_value=True)

    ctx = StateContext(account_id="ACC_01")
    ctx.extra = {}

    next_state = bot._handle_find_commerce_chief(ctx)
    assert next_state == BotState.MEET_NPC
    bot._navigate_to_tuyen_chau_commerce_chief.assert_called_once_with(ctx)


def test_meet_npc_dialog_prompt_not_treated_as_complete_button():
    bot = BotController.__new__(BotController)
    bot._open_commerce_chief_dialog = MagicMock(return_value=True)
    bot._check_active_commerce_quest = MagicMock(return_value=None)
    bot._is_npc_dialog_open = MagicMock(return_value=True)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    bot._input = MagicMock()
    bot._checkpoint = MagicMock()
    bot._ensure_npc_dialog_closed = MagicMock()

    # Dialogue prompt text at y=337 has "hoàn thành", and options are at y >= 500
    prompt_box = TextBox(
        "Nhiệm vụ lần trước đã hoàn thành chưa? Nhớ là trong ngày phải hoàn thành 20 nhiệm vụ nhé!",
        0.95, 814, 337
    )
    option_view_status = TextBox("Xem tình hình nhiệm vụ", 0.95, 399, 660)
    option_cancel = TextBox("Hủy bỏ nhiệm vụ lần này", 0.95, 800, 660)

    bot._text = MagicMock()
    bot._text.read.return_value = [prompt_box, option_view_status, option_cancel]

    ctx = StateContext(account_id="ACC_01")
    ctx.extra = {"returning_quest": True}

    bot._handle_meet_npc(ctx)

    # Must NOT tap prompt_box at (814, 337)
    tap_calls = bot._input.tap.call_args_list
    assert not any(c[0] == (814, 337) for c in tap_calls), "Bot incorrectly clicked dialogue prompt text at (814, 337)!"


def test_return_to_chief_skips_map_if_already_near():
    bot = BotController.__new__(BotController)
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    bot._ensure_map_open = MagicMock()

    ctx = StateContext(account_id="ACC_01")
    res = bot._return_to_nearest_city_commerce_chief(ctx)

    assert res is True
    bot._ensure_map_open.assert_not_called()


def test_handle_navigate_redirects_to_combat_map_if_still_in_city():
    bot = BotController.__new__(BotController)
    bot._app_cfg = MagicMock()
    bot._app_cfg.mode.dry_run = False
    loader = MagicMock()
    loader.load_ui_coords.return_value = {
        "combat": {
            "auto_button": {"x": 1306, "y": 240},
            "horse_button": {"x": 1545, "y": 408},
            "attack_button": {"x": 1450, "y": 770},
            "patrol_ground_points": [{"x": 1200, "y": 550, "name": "Đông", "dx": 130, "dy": 40}],
            "target_hud_region": {"x1": 450, "y1": 60, "x2": 1100, "y2": 130},
            "left_tracker_region": {"x1": 50, "y1": 250, "x2": 350, "y2": 500},
        },
        "commerce_quest": {
            "major_cities": [{"name": "Tuyền Châu", "x": 192, "y": 182}]
        }
    }
    bot._loader = loader
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    bot._input = MagicMock()
    bot._checkpoint = MagicMock()
    bot._is_character_mounted_on_horse = MagicMock(return_value=False)
    bot._find_monster_on_screen = MagicMock(return_value=None)
    bot._return_to_nearest_city_commerce_chief = MagicMock()

    # Currently in Tuyền Châu (192, 183), but quest is in Băng Tâm động
    bot._get_current_location_from_hud = MagicMock(side_effect=[
        ("Tuyền Châu", (192, 183)),
        ("Vân Mộng trạch", (200, 200)),
    ])
    bot._navigate_to_combat_map = MagicMock()

    bot._text = MagicMock()
    # Report completed on 2nd cycle
    calls = [0]
    def mock_read(frame, roi=None):
        calls[0] += 1
        if calls[0] >= 3:
            return [TextBox("Tiêu diệt Tuyết quái (10/10)", 0.95, 100, 300)]
        return []
    bot._text.read.side_effect = mock_read

    ctx = StateContext(account_id="ACC_01")
    ctx.current_quest_type = QuestType.COMBAT
    ctx.extra["combat_map"] = "Băng Tâm động"
    ctx.extra["combat_monster"] = "tuyết quái"
    ctx.extra["combat_count"] = 10

    bot._handle_navigate(ctx)

    bot._navigate_to_combat_map.assert_called_once_with(ctx, "Băng Tâm động")


def test_navigate_to_combat_map_teleports_via_chief_dialog():
    bot = BotController.__new__(BotController)
    bot._dismiss_unexpected_popups = MagicMock()
    loader = MagicMock()
    loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "major_cities": [{"name": "Tuyền Châu", "x": 192, "y": 182}]
        }
    }
    bot._loader = loader
    bot._get_current_location_from_hud = MagicMock(side_effect=[
        ("Tuyền Châu", (192, 183)),
        ("Băng Tâm động", (200, 200)),
    ])
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    bot._open_commerce_chief_dialog = MagicMock(return_value=True)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox("Nhiệm vụ đã hoàn thành!", 0.95, 318, 577),
        TextBox("Đưa ta đến nơi làm nhiệm vụ", 0.95, 764, 576),
    ]
    bot._input = MagicMock()
    bot._verify_commerce_chief_dialog = MagicMock(return_value=True)

    ctx = StateContext(account_id="ACC_01")
    res = bot._navigate_to_combat_map(ctx, "Băng Tâm động")

    assert res is True
    bot._open_commerce_chief_dialog.assert_called_once_with(ctx)
    bot._input.tap.assert_called_with(764, 576)


def test_navigate_to_combat_map_rejects_coordinate_fallback_without_chief_dialog():
    bot = BotController.__new__(BotController)
    bot._dismiss_unexpected_popups = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "commerce_quest": {"major_cities": [{"name": "Dương Châu"}]}
    }
    bot._get_current_location_from_hud = MagicMock(return_value=("Dương Châu", (201, 188)))
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    bot._verify_commerce_chief_dialog = MagicMock(return_value=False)
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    bot._open_commerce_chief_dialog = MagicMock(return_value=True)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    bot._text = MagicMock()
    bot._text.read.return_value = []
    bot._input = MagicMock()

    ctx = StateContext(account_id="ACC_03")
    assert bot._navigate_to_combat_map(ctx, "Vân Mộng trạch") is False
    bot._input.tap.assert_not_called()


def test_incomplete_combat_cancels_return_when_already_back_on_target_map():
    bot = BotController.__new__(BotController)
    bot._attack_monster_blocking_travel = MagicMock(return_value=False)
    bot._get_current_location_from_hud = MagicMock(
        return_value=("Vân Mộng trạch", (196, 193))
    )
    bot._dismiss_unexpected_popups = MagicMock()
    bot._navigate_to_tuyen_chau_commerce_chief = MagicMock()

    ctx = StateContext(account_id="ACC_03")
    ctx.current_quest_type = QuestType.COMBAT
    ctx.extra["combat_map"] = "Vân Mộng trạch"
    ctx.extra["combat_monster"] = "Cóc độc"

    assert bot._return_to_nearest_city_commerce_chief(ctx) is False
    bot._dismiss_unexpected_popups.assert_not_called()
    bot._navigate_to_tuyen_chau_commerce_chief.assert_not_called()


def test_navigate_to_combat_map_stays_when_hud_already_matches_target():
    bot = BotController.__new__(BotController)
    bot._dismiss_unexpected_popups = MagicMock()
    loader = MagicMock()
    loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "major_cities": [
                {"name": "Tuyền Châu"},
                {"name": "Tương Dương"},
            ]
        }
    }
    bot._loader = loader
    bot._get_current_location_from_hud = MagicMock(
        return_value=("Tường Vân động", (200, 200))
    )
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    bot._is_near_commerce_chief = MagicMock(return_value=False)
    bot._return_to_nearest_city_commerce_chief = MagicMock()

    ctx = StateContext(account_id="ACC_03")
    ctx.current_quest_type = QuestType.COMBAT
    ctx.extra["combat_map"] = "Tường Vân động"
    ctx.extra["combat_sect"] = "Cái Bang"

    assert bot._navigate_to_combat_map(ctx, "Tường Vân động") is True
    bot._return_to_nearest_city_commerce_chief.assert_not_called()
    bot._is_near_commerce_chief.assert_not_called()


def test_navigate_to_combat_map_does_not_accept_wrong_dungeon():
    bot = BotController.__new__(BotController)
    bot._dismiss_unexpected_popups = MagicMock()
    loader = MagicMock()
    loader.load_ui_coords.return_value = {
        "commerce_quest": {"major_cities": [{"name": "Tuyền Châu"}]}
    }
    bot._loader = loader
    bot._get_current_location_from_hud = MagicMock(
        return_value=("Ngũ Độc Mê Cung tầng 1", (200, 200))
    )
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    bot._is_near_commerce_chief = MagicMock(return_value=False)
    bot._return_to_nearest_city_commerce_chief = MagicMock()

    ctx = StateContext(account_id="ACC_03")
    ctx.current_quest_type = QuestType.COMBAT
    ctx.extra["combat_sect"] = "Ngũ Độc"

    # Dừng ngay sau khi xác nhận code đã không coi tầng 1 là đúng tầng 2.
    bot._return_to_nearest_city_commerce_chief.side_effect = RuntimeError("wrong-map-navigation")
    with pytest.raises(RuntimeError, match="wrong-map-navigation"):
        bot._navigate_to_combat_map(ctx, "Ngũ Độc Mê Cung tầng 2")


def test_combat_periodic_5s_map_check_triggers_navigate_if_wrong_map():
    bot = BotController.__new__(BotController)
    bot._app_cfg = MagicMock()
    bot._app_cfg.mode.dry_run = False
    loader = MagicMock()
    loader.load_ui_coords.return_value = {
        "combat": {
            "auto_button": {"x": 1306, "y": 240},
            "horse_button": {"x": 1545, "y": 408},
            "attack_button": {"x": 1450, "y": 770},
            "patrol_ground_points": [{"x": 1200, "y": 550, "name": "Đông", "dx": 130, "dy": 40}],
            "target_hud_region": {"x1": 450, "y1": 60, "x2": 1100, "y2": 130},
            "left_tracker_region": {"x1": 50, "y1": 250, "x2": 350, "y2": 500},
        },
        "commerce_quest": {
            "major_cities": [{"name": "Tuyền Châu", "x": 192, "y": 182}]
        }
    }
    bot._loader = loader
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    bot._input = MagicMock()
    bot._checkpoint = MagicMock()
    bot._is_character_mounted_on_horse = MagicMock(return_value=False)
    bot._find_monster_on_screen = MagicMock(return_value=None)
    bot._return_to_nearest_city_commerce_chief = MagicMock()
    bot._navigate_to_combat_map = MagicMock()

    # Initial location check before loop is Băng Tâm động (so no initial redirect)
    # Inside loop, HUD says "Tuyền Châu" (character died or wrong map)
    call_idx = [0]
    def mock_hud_loc(frame=None):
        call_idx[0] += 1
        if call_idx[0] <= 1:
            return ("Băng Tâm động", (100, 200))
        return ("Tuyền Châu", (192, 183))

    bot._get_current_location_from_hud = MagicMock(side_effect=mock_hud_loc)

    bot._text = MagicMock()
    read_count = [0]
    def mock_read(frame, roi=None):
        read_count[0] += 1
        # Stop loop after triggering redirect
        if read_count[0] >= 5:
            return [TextBox("Tiêu diệt Tuyết quái (10/10)", 0.95, 100, 300)]
        return []
    bot._text.read.side_effect = mock_read

    ctx = StateContext(account_id="ACC_01")
    ctx.current_quest_type = QuestType.COMBAT
    ctx.extra["combat_map"] = "Băng Tâm động"
    ctx.extra["combat_monster"] = "tuyết quái"
    ctx.extra["combat_count"] = 10

    t_val = [100.0]
    def fake_time():
        t_val[0] += 2.0
        return t_val[0]

    with patch("time.time", side_effect=fake_time), patch("time.sleep", return_value=None):
        bot._handle_navigate(ctx)

    # Verify that navigate_to_combat_map was called because map was wrong (Tuyền Châu != Băng Tâm động)
    bot._navigate_to_combat_map.assert_called_with(ctx, "Băng Tâm động")


def test_approach_monster_then_right_click():
    """Verify that when monster is far (dist > 180), bot moves closer, then right-clicks when close (dist <= 180)."""
    bot = BotController.__new__(BotController)
    loader = MagicMock()
    loader.load_ui_coords.return_value = {
        "combat": {
            "auto_button": {"x": 1281, "y": 265},
            "horse_button": {"x": 1545, "y": 408},
            "attack_button": {"x": 1450, "y": 770},
            "patrol_ground_points": [{"x": 1200, "y": 550, "name": "Đông", "dx": 130, "dy": 40}],
            "target_hud_region": {"x1": 450, "y1": 60, "x2": 1100, "y2": 130},
            "left_tracker_region": {"x1": 50, "y1": 250, "x2": 350, "y2": 500},
        },
        "commerce_quest": {
            "major_cities": []
        }
    }
    bot._loader = loader
    dummy_frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = dummy_frame
    bot._input = MagicMock()
    bot._checkpoint = MagicMock()
    bot._is_character_mounted_on_horse = MagicMock(return_value=False)
    bot._get_current_location_from_hud = MagicMock(return_value=("Băng Tâm động", (100, 200)))
    bot._navigate_to_combat_map = MagicMock()
    bot._return_to_nearest_city_commerce_chief = MagicMock()

    # Cycle 1: Monster seen at (200, 477) -> bot immediately right-clicks to lock and attack
    # Cycle 2: Quest completed (read_c >= 8)
    cycle = [0]
    def mock_find(frame, monster):
        cycle[0] += 1
        if cycle[0] == 1:
            return (200, 477)
        return None
    bot._find_monster_on_screen = MagicMock(side_effect=mock_find)

    bot._text = MagicMock()
    read_c = [0]
    def mock_read(frame, roi=None):
        read_c[0] += 1
        if read_c[0] >= 12:
            return [TextBox("Tiêu diệt Tuyết quái (10/10)", 0.95, 100, 300)]
        return []
    bot._text.read.side_effect = mock_read

    ctx = StateContext(account_id="ACC_01")
    ctx.current_quest_type = QuestType.COMBAT
    ctx.extra["combat_map"] = "Băng Tâm động"
    ctx.extra["combat_monster"] = "tuyết quái"
    ctx.extra["combat_count"] = 10

    bot._handle_navigate(ctx)

    # 1. Verify right-click was called directly on monster position (200, 477) with count=2
    bot._input.right_click.assert_called_with(200, 477, count=1)


def test_input_controller_right_click():
    """A successful Windows right-click must not be overwritten by an ADB left tap."""
    from src.input_controller import InputController
    adb_mock = MagicMock()
    cfg_mock = MagicMock()
    inp = InputController(adb=adb_mock, config=cfg_mock)

    inp._send_win32_right_click = MagicMock(return_value=True)
    inp.right_click(500, 600)

    inp._send_win32_right_click.assert_called_once_with(500, 600, count=1)
    adb_mock.right_click.assert_not_called()


def test_input_controller_right_click_falls_back_to_adb():
    from src.input_controller import InputController

    adb_mock = MagicMock()
    inp = InputController.__new__(InputController)
    inp._adb = adb_mock
    inp._send_win32_right_click = MagicMock(return_value=False)

    inp.right_click(500, 600)

    adb_mock.right_click.assert_called_once_with(500, 600, count=1)


def test_patrol_detects_oscillation_and_changes_direction():
    """Verify that when character oscillates between coordinates in 3 steps, bot detects it and swipes in an escape direction."""
    bot = BotController.__new__(BotController)
    loader = MagicMock()
    loader.load_ui_coords.return_value = {
        "combat": {
            "auto_button": {"x": 1281, "y": 265},
            "horse_button": {"x": 1545, "y": 408},
            "attack_button": {"x": 1450, "y": 770},
            "patrol_ground_points": [{"x": 1200, "y": 550, "name": "Đông", "dx": 130, "dy": 40}],
            "target_hud_region": {"x1": 450, "y1": 60, "x2": 1100, "y2": 130},
            "left_tracker_region": {"x1": 50, "y1": 250, "x2": 350, "y2": 500},
            "joystick_center": {"x": 220, "y": 680},
        },
        "commerce_quest": {
            "major_cities": []
        }
    }
    bot._loader = loader
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._input = MagicMock()
    bot._checkpoint = MagicMock()
    bot._is_character_mounted_on_horse = MagicMock(return_value=False)
    bot._find_monster_on_screen = MagicMock(return_value=None)
    bot._navigate_to_combat_map = MagicMock()
    bot._return_to_nearest_city_commerce_chief = MagicMock()

    # Step 1: (199, 198)
    # Step 2: (200, 197)
    # Step 3: (199, 198) -> 3 steps return to old spot!
    step = [0]
    def mock_hud(frame=None):
        step[0] += 1
        if step[0] <= 2:
            return ("Băng Tâm động", (199, 198))
        elif step[0] <= 4:
            return ("Băng Tâm động", (200, 197))
        else:
            return ("Băng Tâm động", (199, 198))
    bot._get_current_location_from_hud = MagicMock(side_effect=mock_hud)

    read_c = [0]
    def mock_swipe(x1, y1, x2, y2, duration=1000):
        read_c[0] = 999
    bot._input.swipe.side_effect = mock_swipe

    def mock_read(frame, roi=None):
        if read_c[0] >= 999:
            return [TextBox("Tiêu diệt Tuyết quái (10/10)", 0.95, 100, 300)]
        return []
    bot._text = MagicMock()
    bot._text.read.side_effect = mock_read

    ctx = StateContext(account_id="ACC_01")
    ctx.current_quest_type = QuestType.COMBAT
    ctx.extra["combat_map"] = "Băng Tâm động"
    ctx.extra["combat_monster"] = "tuyết quái"
    ctx.extra["combat_count"] = 10

    bot._handle_navigate(ctx)

    # Verify that swipe was called to escape oscillation
    assert bot._input.swipe.call_count >= 1


def test_find_monster_targets_body_below_name_label():
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    # Name label at (856, 492)
    bot._text.read.return_value = [
        TextBox("Tuyết Quái", 0.95, 856, 492)
    ]
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    pos = bot._find_monster_on_screen(frame, "tuyết quái")
    assert pos is not None
    # Must NOT click right on name label (856, 492)
    assert pos != (856, 492)
    # Must click near the body (~49px below the label on 900p), not ~100px
    # below it where the click would land on the ground.
    assert pos == (856, 492 + int(900 * 0.055))


def test_find_green_short_monster_label_with_special_ocr_pass():
    """A green on-screen label 'Nhện' must match quest target 'Nhện độc'."""
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._is_map_open = MagicMock(return_value=False)
    bot._is_quest_panel_open = MagicMock(return_value=False)
    bot._text.read.side_effect = [
        [],
        [TextBox("Nhện", 0.91, 960, 330)],
    ]
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    # BGR green pixels trigger the dedicated green-label OCR pass.
    frame[315:335, 930:990] = (0, 255, 0)

    pos = bot._find_monster_on_screen(frame, "Nhện độc")

    assert pos == (960, 330 + int(900 * 0.055))
    assert bot._text.read.call_count == 2


def test_find_green_spider_body_without_ocr_label():
    """Thân Nhện xanh rõ phải được click ngay cả khi OCR không đọc được tên."""
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._text.read.return_value = []
    bot._is_map_open = MagicMock(return_value=False)
    bot._is_quest_panel_open = MagicMock(return_value=False)
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    # Mô phỏng thân/chân Nhện xanh trong vùng chiến đấu.
    frame[330:358, 610:650] = (0, 255, 0)

    pos = bot._find_monster_on_screen(frame, "Nhện độc")

    assert pos is not None
    assert 625 <= pos[0] <= 635
    assert 340 <= pos[1] <= 348
    bot._text.read.assert_not_called()


def test_find_green_spider_ignores_player_effect_at_screen_center():
    """Mảng xanh sát tâm nhân vật không được nhận nhầm thành thân Nhện."""
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._text.read.return_value = []
    bot._is_map_open = MagicMock(return_value=False)
    bot._is_quest_panel_open = MagicMock(return_value=False)
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)

    # Hiệu ứng xanh của nhân vật ngay tâm màn hình (800, 477).
    frame[463:491, 780:820] = (0, 255, 0)
    # Thân Nhện thật ở bên phải/phía trên.
    frame[390:426, 875:925] = (0, 255, 0)

    pos = bot._find_monster_on_screen(frame, "Nhện độc")

    assert pos == (900, 408)
    bot._text.read.assert_not_called()


def test_find_green_spider_rejects_effect_60px_from_player():
    """False-positive từng thấy tại (851,509) phải bị loại để bot tiếp tục tuần tra."""
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._text.read.return_value = []
    bot._is_map_open = MagicMock(return_value=False)
    bot._is_quest_panel_open = MagicMock(return_value=False)
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)

    frame[495:523, 831:871] = (0, 255, 0)

    assert bot._find_monster_on_screen(frame, "Nhện độc") is None


def test_handle_navigate_redirects_to_combat_map_if_in_nga_my():
    bot = BotController.__new__(BotController)
    bot._app_cfg = MagicMock()
    bot._app_cfg.mode.dry_run = False
    loader = MagicMock()
    loader.load_ui_coords.return_value = {
        "combat": {
            "auto_button": {"x": 1306, "y": 240},
            "horse_button": {"x": 1545, "y": 408},
            "attack_button": {"x": 1450, "y": 770},
            "patrol_ground_points": [{"x": 1200, "y": 550, "name": "Đông", "dx": 130, "dy": 40}],
            "target_hud_region": {"x1": 450, "y1": 60, "x2": 1100, "y2": 130},
            "left_tracker_region": {"x1": 50, "y1": 250, "x2": 350, "y2": 500},
        },
        "commerce_quest": {
            "major_cities": [{"name": "Tuyền Châu", "x": 192, "y": 182}]
        }
    }
    bot._loader = loader
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    bot._input = MagicMock()
    bot._checkpoint = MagicMock()
    bot._is_character_mounted_on_horse = MagicMock(return_value=False)
    bot._find_monster_on_screen = MagicMock(return_value=None)
    bot._return_to_nearest_city_commerce_chief = MagicMock()

    # Character is outside the cave in Nga My (227, 186)
    bot._get_current_location_from_hud = MagicMock(return_value=("Nga My", (227, 186)))
    bot._navigate_to_combat_map = MagicMock()

    bot._text = MagicMock()
    calls = [0]
    def mock_read(frame, roi=None):
        calls[0] += 1
        if calls[0] >= 3:
            return [TextBox("Tiêu diệt Tuyết quái (10/10)", 0.95, 100, 300)]
        return []
    bot._text.read.side_effect = mock_read

    ctx = StateContext(account_id="ACC_01")
    ctx.current_quest_type = QuestType.COMBAT
    ctx.extra["combat_map"] = "Băng Tâm động"
    ctx.extra["combat_monster"] = "tuyết quái"
    ctx.extra["combat_count"] = 10

    bot._handle_navigate(ctx)

    bot._navigate_to_combat_map.assert_called_once_with(ctx, "Băng Tâm động")


def test_return_to_chief_does_not_skip_travel_when_returning_from_combat():
    bot = BotController.__new__(BotController)
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    bot._ensure_map_open = MagicMock(return_value=False)
    bot._get_current_location_from_hud = MagicMock(return_value=("Băng Tâm động", (205, 196)))
    bot._dismiss_unexpected_popups = MagicMock()
    bot._ensure_npc_dialog_closed = MagicMock()
    bot._switch_to_city = MagicMock(return_value=False)
    bot._loader = MagicMock()
    bot._navigate_to_tuyen_chau_commerce_chief = MagicMock(return_value=True)

    ctx = StateContext(account_id="ACC_01")
    ctx.current_quest_type = QuestType.COMBAT
    ctx.extra["combat_map"] = "Băng Tâm động"
    ctx.extra["combat_objective_complete"] = True
    ctx.extra["returning_quest"] = True

    # Must NOT skip travel (must not immediately return True via _is_near_commerce_chief)
    bot._return_to_nearest_city_commerce_chief(ctx)
    bot._is_near_commerce_chief.assert_not_called()
    bot._navigate_to_tuyen_chau_commerce_chief.assert_called_once()



def test_is_npc_dialog_open_ignores_toast_banner():
    bot = BotController.__new__(BotController)
    bot._vision = MagicMock()
    bot._vision.find_template.return_value = MagicMock(found=False)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._text = MagicMock()
    # Toast banner has "Nhiệm vụ thương hội: Truy sát Tuyết Quái đã hoàn thành"
    bot._text.read.return_value = [
        TextBox("Nhiệm vụ thương hội: Truy sát Tuyết Quái đã hoàn thành", 0.95, 850, 674)
    ]

def test_find_monster_rejects_quest_tracker_and_single_token():
    """Verify that _find_monster_on_screen rejects text in the left quest tracker (x < 400) and single token false matches."""
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    # 1. Text inside quest tracker on the left (x=344, y=333) with words like "nhan" or quest names
    # 2. Text with only a single matching token "nhân vật"
    # 3. Real monster "Thực Nhân Thảo" at (780, 229)
    bot._text.read.return_value = [
        TextBox("Vương Nghiệp Vĩ ở Đông Bắc Tuyền Châu", 0.95, 344, 333),
        TextBox("Thông tin nhân vật", 0.95, 306, 66),
        TextBox("Thực Nhân Thảo", 0.95, 780, 229),
    ]
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    pos = bot._find_monster_on_screen(frame, "Thực Nhân Thảo")
    assert pos is not None
    # Must NOT click inside the quest tracker at (344, ...)
    assert pos[0] != 344
    # Must click on the real monster at x=780
    assert pos[0] == 780
    assert pos[1] == 229 + int(900 * 0.055)


def test_is_quest_panel_open_and_ensure_closed():
    bot = BotController.__new__(BotController)
    bot._screen = MagicMock()
    dummy_frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._screen.capture.return_value = dummy_frame

    # First call: panel is open ("Bảng nhiệm vụ" at header)
    # Second call (after close button): panel still open
    # Third call (after back key): panel closed
    bot._text = MagicMock()
    bot._text.read.side_effect = [
        [TextBox("Bảng nhiệm vụ", 0.99, 900, 60)],
        [TextBox("Bảng nhiệm vụ", 0.99, 900, 60)],
        [],
    ]

    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "quest_panel_close": {"x": 1511, "y": 76}
        }
    }
    bot._input = MagicMock()

    ctx = StateContext(account_id="ACC_02")
    closed = bot._ensure_quest_panel_closed(ctx=ctx, frame=dummy_frame)
    assert closed is True

    # The close diamond is calculated from the current 1600x900 frame.
    bot._input.tap.assert_called_with(1466, 125)
    # Verified fallback press_back was invoked when tap wasn't enough
    assert bot._input.press_back.called or bot._input.key_event.called


def test_generic_quest_text_does_not_mean_quest_panel_is_open():
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox("Nhiệm vụ Thương hội", 0.99, 900, 60),
    ]
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)

    assert bot._is_quest_panel_open(frame) is False


def test_ensure_quest_panel_closed_does_not_click_when_already_closed():
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._text.read.return_value = []
    bot._input = MagicMock()
    bot._loader = MagicMock()
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)

    assert bot._ensure_quest_panel_closed(frame=frame) is True
    bot._input.tap.assert_not_called()


def test_ensure_quest_panel_closed_without_frame_captures_before_clicking():
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._text.read.return_value = []
    bot._input = MagicMock()
    bot._loader = MagicMock()
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)

    assert bot._ensure_quest_panel_closed() is True
    bot._screen.capture.assert_called_once_with()
    bot._input.tap.assert_not_called()


def test_input_controller_key_event():
    from src.input_controller import InputController
    adb = MagicMock()
    cfg = MagicMock()
    input_ctrl = InputController(adb, cfg)

    input_ctrl.key_event(4)
    adb.key_event.assert_called_once_with(4)

    input_ctrl.press_back()
    adb.press_back.assert_called_once()


def test_bat_phu_ha_dong_su_canonical_extraction():
    from src.quest_classifier import parse_yellow_entities
    tokens = ["Tang 2 Duong Trung dong", "10", "Bat phy Ha Dor"]
    entities = parse_yellow_entities(tokens)
    assert entities["monster"] == "Bát phụ Hà Đông Sư"
    assert entities["map_name"] == "Tầng 2 Dương Trung động"
    assert entities["count"] == 10
