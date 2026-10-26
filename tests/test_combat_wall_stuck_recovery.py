from unittest.mock import MagicMock, call
import numpy as np
import pytest

from src.bot_controller import BotController
from src.state_machine import StateContext, QuestType, BotState
from src.text_recognition import TextBox


def test_patrol_points_no_north_vectors():
    """Verify that all default patrol points have dy >= 0 to avoid hitting the north cliff wall."""
    from src.config_loader import ConfigLoader
    loader = ConfigLoader()
    coords = loader.load_ui_coords()
    combat = coords.get("combat", {})
    patrol_pts = combat.get("patrol_ground_points", [])

    assert len(patrol_pts) > 0, "patrol_ground_points should not be empty"
    for pt in patrol_pts:
        # None of the patrol points should head north into the cliff
        dy = pt.get("dy", 0)
        assert dy >= 0, f"Patrol point {pt.get('name')} has negative dy {dy}, will hit north cliff wall!"
        assert pt.get("y", 500) >= 440, f"Patrol point y={pt.get('y')} is too close to north wall!"


def test_wall_stuck_detection_and_escape():
    """Verify that if coordinates don't change for 2 cycles, bot triggers unstuck escape action."""
    bot = BotController.__new__(BotController)
    loader = MagicMock()
    loader.load_ui_coords.return_value = {
        "combat": {
            "auto_button": {"x": 1306, "y": 240},
            "horse_button": {"x": 1545, "y": 408},
            "attack_button": {"x": 1450, "y": 770},
            "patrol_ground_points": [
                {"x": 1200, "y": 550, "name": "Đông", "dx": 130, "dy": 40}
            ],
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
    bot._in_combat = False

    dummy_frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._screen.capture.return_value = dummy_frame

    # Simulate: No monster found, coordinates stuck at (199, 196)
    bot._find_monster_on_screen = MagicMock(return_value=None)
    bot._get_current_location_from_hud = MagicMock(return_value=("Bang Tam dong1", (199, 196)))
    bot._is_character_mounted_on_horse = MagicMock(return_value=False)
    bot._ensure_npc_dialog_closed = MagicMock()

    # Tracker text: not complete on first 3 cycles, then complete on 4th
    cycle_count = {"val": 0}
    def mock_read(frame, region=None):
        if region == (50, 250, 350, 500): # tracker_roi
            cycle_count["val"] += 1
            if cycle_count["val"] >= 4:
                return [TextBox("Tieu diet Tuyet Quai 10/10", 0.9, 150, 300)]
            return [TextBox("Tieu diet Tuyet Quai 0/10", 0.9, 150, 300)]
        return []

    bot._text.read.side_effect = mock_read
    bot._check_quest_panel = MagicMock(return_value=False)
    bot._return_to_nearest_city_commerce_chief = MagicMock()

    ctx = StateContext(account_id="acc1")
    ctx.current_quest_type = QuestType.COMBAT
    ctx.current_quest_detail = {"monster": "Tuyet Quai", "count": 10, "map_name": "Bang Tam dong1"}
    ctx.extra["combat_map"] = "Bang Tam dong1"

    res = bot._handle_navigate(ctx)

    # Verify that unstuck swipe was called
    swipe_calls = bot._input.swipe.call_args_list
    unstuck_swipes = [c for c in swipe_calls if c[1].get("duration") in (1200, 1500)]
    assert len(unstuck_swipes) >= 1, f"Expected unstuck swipe call, but got {swipe_calls}"


def test_monster_behind_wall_recovery():
    """Verify that when targeting a monster behind a wall and coordinates don't move, bot turns off auto, blacklists monster, and walks along wall."""
    bot = BotController.__new__(BotController)
    loader = MagicMock()
    loader.load_ui_coords.return_value = {
        "combat": {
            "auto_button": {"x": 1306, "y": 240},
            "horse_button": {"x": 1545, "y": 408},
            "attack_button": {"x": 1450, "y": 770},
            "patrol_ground_points": [
                {"x": 1200, "y": 550, "name": "Đông", "dx": 130, "dy": 40}
            ],
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
    bot._in_combat = False

    dummy_frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._screen.capture.return_value = dummy_frame

    # Character stuck at wall (215, 215) in Ngũ Độc mê cung 2
    bot._get_current_location_from_hud = MagicMock(return_value=("Ngũ Độc mê cung 2", (215, 215)))
    bot._is_character_mounted_on_horse = MagicMock(return_value=False)
    bot._ensure_npc_dialog_closed = MagicMock()

    # Track calls to _find_monster_on_screen
    call_records = []
    def mock_find_monster(frame, monster_name, blocked_positions=None):
        call_records.append({"blocked": list(blocked_positions) if blocked_positions else []})
        # If (481, 211) is blocked, return None (monster ignored)
        if blocked_positions:
            for bp in blocked_positions:
                pos = bp.get("pos", (0, 0))
                if abs(pos[0] - 481) < 50 and abs(pos[1] - 211) < 50:
                    return None
        return (481, 211)

    bot._find_monster_on_screen = MagicMock(side_effect=mock_find_monster)

    # Tracker text: complete after 5 cycles
    cycle_count = {"val": 0}
    def mock_read(frame, region=None):
        if region == (50, 250, 350, 500): # tracker_roi
            cycle_count["val"] += 1
            if cycle_count["val"] >= 5:
                return [TextBox("Tieu diet Thuc Nhan Thao 10/10", 0.9, 150, 300)]
            return [TextBox("Tieu diet Thuc Nhan Thao 0/10", 0.9, 150, 300)]
        return []

    bot._text.read.side_effect = mock_read
    bot._check_quest_panel = MagicMock(return_value=False)
    bot._return_to_nearest_city_commerce_chief = MagicMock()

    ctx = StateContext(account_id="acc1")
    ctx.current_quest_type = QuestType.COMBAT
    ctx.current_quest_detail = {"monster": "Thuc Nhan Thao", "count": 10, "map_name": "Ngũ Độc Mê Cung tầng 2"}

    res = bot._handle_navigate(ctx)

    # Check that wall follow swipe was triggered:
    # WALL_FOLLOW_VECTORS first vector is ("Đông Nam...", 140, 120, ...) -> joy_x+140=360, joy_y+120=800
    swipe_calls = bot._input.swipe.call_args_list
    wall_follow_swipes = [c for c in swipe_calls if c[0][2] == 360 and c[0][3] == 800 and c[1].get("duration") == 3000]
    assert len(wall_follow_swipes) >= 1, f"Expected wall follow swipe (360, 800), got: {swipe_calls}"

    # Verify that in subsequent calls, blocked_positions was passed with (481, 211)
    passed_blocked = [r for r in call_records if any(bp.get("pos") == (481, 211) for bp in r["blocked"])]
    assert len(passed_blocked) >= 1, f"Expected (481, 211) to be in blocked_positions, got records: {call_records}"


def test_is_same_combat_map_floor_discrimination():
    """Verify that _is_same_combat_map distinguishes between different floors of the same dungeon."""
    bot = BotController.__new__(BotController)

    # Different floors MUST NOT match
    assert bot._is_same_combat_map("Ngũ Độc Mê Cung tầng 1", "Ngũ Độc Mê Cung tầng 2", "Ngũ Độc") is False
    assert bot._is_same_combat_map("Ngu Doc me cung tang1", "Ngũ Độc Mê Cung tầng 2", "Ngũ Độc") is False
    assert bot._is_same_combat_map("Ngũ Độc mê cung 1", "Ngũ Độc mê cung 2", "Ngũ Độc") is False

    # Same floor MUST match
    assert bot._is_same_combat_map("Ngũ Độc mê cung 2", "Ngũ Độc Mê Cung tầng 2", "Ngũ Độc") is True
    assert bot._is_same_combat_map("Ngi Doc me cung2", "Ngũ Độc Mê Cung tầng 2", "Ngũ Độc") is True
    assert bot._is_same_combat_map("Ngũ Độc Mê Cung tầng 1", "Ngũ Độc Mê Cung tầng 1", "Ngũ Độc") is True

    # Cave / Non-floor maps
    assert bot._is_same_combat_map("Tường Vân động", "Tường Vân động", "Cái Bang") is True
    assert bot._is_same_combat_map("Biện Kinh", "Ngũ Độc Mê Cung tầng 2", "Ngũ Độc") is False


def test_major_city_is_never_fuzzy_matched_to_combat_map():
    """Regression: Dương Châu must not be treated as Vân Mộng trạch."""
    bot = BotController.__new__(BotController)

    assert bot._is_same_combat_map("Dương Châu", "Vân Mộng trạch") is False
    assert bot._is_same_combat_map("Duong Chau", "Van Mong trach") is False


def test_wrong_floor_detection_triggers_renavigation():
    """Verify that when character wanders into wrong floor (e.g. tầng 1 instead of tầng 2), bot detects it and triggers re-navigation."""
    bot = BotController.__new__(BotController)
    loader = MagicMock()
    loader.load_ui_coords.return_value = {
        "combat": {
            "auto_button": {"x": 1306, "y": 240},
            "horse_button": {"x": 1545, "y": 408},
            "attack_button": {"x": 1450, "y": 770},
            "patrol_ground_points": [{"x": 1200, "y": 550, "name": "Đông", "dx": 130, "dy": 40}],
            "target_hud_region": {"x1": 450, "y1": 60, "x2": 1100, "y2": 130},
            "left_tracker_region": {"x1": 50, "y1": 250, "x2": 350, "y2": 500},
        }
    }
    bot._loader = loader
    bot._input = MagicMock()
    bot._screen = MagicMock()
    bot._text = MagicMock()
    bot._checkpoint = MagicMock()
    bot._in_combat = False

    dummy_frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._screen.capture.return_value = dummy_frame

    # Cycle 1: Character is at tầng 1 (194, 198) when quest requires tầng 2!
    # Cycle 2: After re-navigation, arrives at tầng 2 (215, 215) and finishes quest
    hud_calls = [0]
    def mock_hud(frame=None):
        hud_calls[0] += 1
        if hud_calls[0] <= 2:
            return ("Ngũ Độc Mê Cung tầng 1", (194, 198))
        return ("Ngũ Độc Mê Cung tầng 2", (215, 215))

    bot._get_current_location_from_hud = MagicMock(side_effect=mock_hud)
    bot._is_character_mounted_on_horse = MagicMock(return_value=False)
    bot._find_monster_on_screen = MagicMock(return_value=None)
    bot._navigate_to_combat_map = MagicMock()
    bot._ensure_npc_dialog_closed = MagicMock()

    # Tracker completes on 4th read
    read_c = [0]
    def mock_read(frame, roi=None):
        if roi == (50, 250, 350, 500):
            read_c[0] += 1
            if read_c[0] >= 3:
                return [TextBox("Tieu diet Thuc Nhan Thao 10/10", 0.9, 150, 300)]
        return []
    bot._text.read.side_effect = mock_read
    bot._check_quest_panel = MagicMock(return_value=False)
    bot._return_to_nearest_city_commerce_chief = MagicMock()

    ctx = StateContext(account_id="ACC_01")
    ctx.current_quest_type = QuestType.COMBAT
    ctx.extra["combat_map"] = "Ngũ Độc Mê Cung tầng 2"
    ctx.extra["combat_monster"] = "Thuc Nhan Thao"
    ctx.extra["combat_count"] = 10
    ctx.extra["combat_sect"] = "Ngũ Độc"

    res = bot._handle_navigate(ctx)

    # Verify that _navigate_to_combat_map was called to bring character back to tầng 2
    bot._navigate_to_combat_map.assert_called_with(ctx, "Ngũ Độc Mê Cung tầng 2")


def test_combat_records_reverse_vector_and_escapes_on_re_entry():
    """Verify that leaving the map saves reverse vector, and entering map swipes in opposite direction."""
    bot = BotController.__new__(BotController)
    loader = MagicMock()
    loader.load_ui_coords.return_value = {
        "combat": {
            "auto_button": {"x": 1306, "y": 240},
            "horse_button": {"x": 1545, "y": 408},
            "attack_button": {"x": 1450, "y": 770},
            "patrol_ground_points": [
                {"x": 1200, "y": 550, "name": "Đông Nam", "dx": 130, "dy": 80}
            ],
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
    bot._in_combat = False

    dummy_frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._screen.capture.return_value = dummy_frame

    # Case 1: Re-entering map with reverse vector (-130, -80)
    ctx = StateContext(account_id="ACC_01")
    ctx.current_quest_type = QuestType.COMBAT
    ctx.extra["combat_map"] = "Băng Tâm động"
    ctx.extra["combat_monster"] = "Tuyết quái"
    ctx.extra["combat_count"] = 10
    ctx.extra["combat_reverse_vector"] = (-130, -80)

    bot._get_current_location_from_hud = MagicMock(return_value=("Băng Tâm động", (200, 200)))
    bot._is_character_mounted_on_horse = MagicMock(return_value=False)
    bot._find_monster_on_screen = MagicMock(return_value=None)
    bot._text.read.return_value = [TextBox("Tieu diet Tuyet quai 10/10", 0.9, 150, 300)]
    bot._check_quest_panel = MagicMock(return_value=False)
    bot._return_to_nearest_city_commerce_chief = MagicMock()

    res = bot._handle_navigate(ctx)

    # Verify joystick was swiped in reverse direction (-130, -80) for 2000ms
    # joy_x=220, joy_y=680 -> target = (220 - 130 = 90, 680 - 80 = 600)
    bot._input.swipe.assert_any_call(220, 680, 90, 600, duration=2000)
    assert "combat_reverse_vector" not in ctx.extra


