import time
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from src.bot_controller import BotController
from src.state_machine import StateContext, QuestType, BotState
from src.text_recognition import TextBox


def test_combat_skills_visible_never_clicks_tuy_chon_from_ocr_labels():
    bot = BotController.__new__(BotController)
    bot._input = MagicMock()
    bot._text = MagicMock()

    # Simulated boxes showing feature menu is open in bottom-right
    box_vc = TextBox("Võ công", 0.95, 1400, 650)
    box_cd = TextBox("Cài đặt", 0.95, 1400, 870)
    box_tc = TextBox("Tùy chọn", 0.95, 969, 882)

    dummy_frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._text.read.return_value = [box_vc, box_cd, box_tc]

    bot._ensure_combat_skills_visible(dummy_frame)
    bot._input.tap.assert_not_called()
    bot._text.read.assert_not_called()


def test_auto_state_recognized_with_ty_danh():
    from src.text_recognition import normalize_game_text
    raw_texts = ["ane.ty.danh...", "Tự đánh", "Đang tự đánh", "Doi", "Tui"]
    is_auto = any(
        any(kw in normalize_game_text(t) for kw in ("tu danh", "ty danh", "dang tu", "dang ty", "tu dan", "ty dan", "danh..."))
        for t in raw_texts
    )
    assert is_auto is True


def test_combat_quest_10min_timeout_triggers_abandon_return():
    bot = BotController.__new__(BotController)
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "combat": {
            "auto_button": {"x": 1308, "y": 241},
            "attack_button": {"x": 1450, "y": 770},
            "horse_button": {"x": 1553, "y": 370},
        }
    }
    bot._input = MagicMock()
    bot._text = MagicMock()
    bot._text.read.return_value = []
    bot._screen = MagicMock()
    dummy_frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._screen.capture.return_value = dummy_frame
    bot._ensure_quest_panel_closed = MagicMock()
    bot._ensure_combat_skills_visible = MagicMock()
    bot._ensure_dismounted_from_horse = MagicMock()
    bot._collapse_chat_bar = MagicMock()
    bot._return_to_nearest_city_commerce_chief = MagicMock(return_value=True)
    bot._get_current_location_from_hud = MagicMock(return_value=("Dương Trung động", (100, 100)))
    bot._is_same_combat_map = MagicMock(return_value=True)
    bot._checkpoint = MagicMock()

    ctx = StateContext(account_id="ACC_02")
    ctx.current_quest_type = QuestType.COMBAT
    ctx.extra = {
        "combat_map": "Dương Trung động",
        "combat_monster": "Bát phụ Hà Đông Sư",
        "combat_count": 10,
        # Started 605 seconds ago (> 10 minutes)
        "combat_quest_start_time": time.time() - 605,
    }

    next_state = bot._handle_navigate(ctx)
    assert next_state == BotState.RETURN_QUEST
    assert ctx.extra.get("abandon_combat_quest") is True
    assert ctx.extra.get("returning_quest") is True
    bot._return_to_nearest_city_commerce_chief.assert_called_once_with(ctx)


def test_handle_meet_npc_abandons_combat_quest():
    bot = BotController.__new__(BotController)
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    bot._is_npc_dialog_open = MagicMock(return_value=True)
    bot._check_active_commerce_quest = MagicMock(return_value=None)
    bot._open_commerce_chief_dialog = MagicMock(return_value=True)
    bot._screen = MagicMock()
    dummy_frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._screen.capture.return_value = dummy_frame
    bot._input = MagicMock()
    bot._ensure_npc_dialog_closed = MagicMock()
    bot._select_contribute_to_receive_quest = MagicMock(return_value=BotState.CLASSIFY_QUEST)

    step1_cancel_option = TextBox("Ta muốn hủy bỏ nhiệm vụ đang làm", 0.95, 1167, 576)
    step2_confirm_btn = TextBox("Hủy bỏ nhiệm vụ", 0.95, 303, 571)
    step3_finish_btn = TextBox("Ket thic doithoaf", 0.95, 309, 543)

    bot._text = MagicMock()
    bot._text.read.side_effect = [
        [step1_cancel_option],  # dialog_boxes read (Step 1)
        [step2_confirm_btn],    # step2 read (Step 2)
        [step3_finish_btn],     # step3 read (Step 3)
    ]

    ctx = StateContext(account_id="ACC_02")
    ctx.current_quest_type = QuestType.COMBAT
    ctx.extra = {
        "abandon_combat_quest": True,
        "combat_map": "Dương Trung động",
        "combat_monster": "Bát phụ Hà Đông Sư",
        "combat_count": 10,
        "combat_quest_start_time": 123456,
        "dismounted_for_combat": True,
    }

    next_state = bot._handle_meet_npc(ctx)
    assert next_state == BotState.CLASSIFY_QUEST
    # Clicked step 1 cancel option button: "Ta muốn hủy bỏ nhiệm vụ đang làm"
    bot._input.tap.assert_any_call(1167, 576)
    # Clicked step 2 confirm button: "Hủy bỏ nhiệm vụ"
    bot._input.tap.assert_any_call(303, 571)
    # Clicked step 3 finish dialogue button: "Kết thúc đối thoại"
    bot._input.tap.assert_any_call(309, 543)
    # Extra flags are cleaned up
    assert "abandon_combat_quest" not in ctx.extra
    assert "combat_map" not in ctx.extra
    assert "combat_quest_start_time" not in ctx.extra
    assert "dismounted_for_combat" not in ctx.extra
    assert ctx.current_quest_type == QuestType.UNKNOWN
    # Called contribute to receive next quest
    bot._select_contribute_to_receive_quest.assert_called_once_with(ctx)


def test_is_forbidden_ui_coordinate_blocks_sensitive_zones():
    bot = BotController.__new__(BotController)

    # 1. Feature dock buttons: chỉ nút Nhiệm vụ được phép, còn lại bị chặn
    assert bot.is_forbidden_ui_coordinate(1450, 770) is False # Nhiệm vụ: ngoại lệ được phép
    assert bot.is_forbidden_ui_coordinate(1450, 770, allow_quest_button=False) is True
    assert bot.is_forbidden_ui_coordinate(1364, 548) is True  # Võ công
    assert bot.is_forbidden_ui_coordinate(1479, 547) is True  # Nhân vật
    assert bot.is_forbidden_ui_coordinate(1364, 866) is True  # Cài đặt
    assert bot.is_forbidden_ui_coordinate(1530, 745) is True  # Tổ đội, sát Nhiệm vụ
    assert bot.is_forbidden_ui_coordinate(1515, 865) is True  # Hảo hữu

    # 2. Top icons (Phúc lợi, Hạng, Ngự các, Mở khóa, Luyện công)
    assert bot.is_forbidden_ui_coordinate(959, 123) is True   # Phúc lợi
    assert bot.is_forbidden_ui_coordinate(1057, 123) is True  # Hạng
    assert bot.is_forbidden_ui_coordinate(1159, 121) is True  # Ngự các
    assert bot.is_forbidden_ui_coordinate(234, 151) is True   # Mở khóa
    assert bot.is_forbidden_ui_coordinate(439, 153) is True   # Luyện công

    # 3. Minimap & Quick dock (Túi, Đổi, Nhặt, Buff, Ngựa)
    assert bot.is_forbidden_ui_coordinate(1432, 52) is True   # Minimap
    assert bot.is_forbidden_ui_coordinate(1350, 344) is True  # Túi / Đổi
    assert bot.is_forbidden_ui_coordinate(1510, 423) is True  # Ngựa

    # 4. Quest banner & Left controls (Tân thủ banner, Linh phách, Thiền)
    assert bot.is_forbidden_ui_coordinate(235, 381) is True   # Tân thủ banner
    assert bot.is_forbidden_ui_coordinate(73, 567) is True    # Linh phách
    assert bot.is_forbidden_ui_coordinate(71, 828) is True    # Thiền

    # 5. Bottom chat & Tùy chọn
    assert bot.is_forbidden_ui_coordinate(200, 850) is True   # Kênh chat
    assert bot.is_forbidden_ui_coordinate(510, 835) is True   # Nút Nhanh giữa-dưới
    assert bot.is_forbidden_ui_coordinate(750, 845) is True   # Cụm chat giữa-dưới
    assert bot.is_forbidden_ui_coordinate(940, 835) is True   # Nút Tùy chọn giữa-dưới
    assert bot.is_forbidden_ui_coordinate(940, 879) is True   # Tùy chọn

    # 6. Safe combat ground coordinates MUST NOT be blocked
    assert bot.is_forbidden_ui_coordinate(800, 500) is False  # Safe ground center
    assert bot.is_forbidden_ui_coordinate(1100, 550) is False # Safe ground east
    assert bot.is_forbidden_ui_coordinate(550, 500) is False  # Safe ground west
    assert bot.is_forbidden_ui_coordinate(800, 700) is False  # Safe ground south
    assert bot.is_forbidden_ui_coordinate(800, 250) is False  # Safe ground north


def test_clamp_to_safe_combat_ground():
    bot = BotController.__new__(BotController)
    # Inside safe ground remains unchanged
    assert bot.clamp_to_safe_combat_ground(800, 500) == (800, 500)

    # Clamping too far left (into quest banner)
    cx, cy = bot.clamp_to_safe_combat_ground(200, 450)
    assert cx >= 448
    assert cy == 450

    # Clamping too far right (into feature dock)
    cx, cy = bot.clamp_to_safe_combat_ground(1500, 700)
    assert cx <= 1184
    assert cy == 700

    # Clamping too far up (into HUD/top icons)
    cx, cy = bot.clamp_to_safe_combat_ground(800, 50)
    assert cx == 800
    assert cy >= 198


def test_combat_patrol_uses_joystick_before_ground_tap():
    bot = BotController.__new__(BotController)
    bot._input = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "combat": {
            "joystick_center": {"x": 220, "y": 680},
            "patrol_joystick_hold_ms": 800,
        }
    }

    point = {"x": 1200, "y": 550, "dx": 130, "dy": 40}
    with patch("src.bot_controller.time.sleep", return_value=None):
        result = bot._move_combat_patrol(point, 1600, 900)

    bot._input.swipe.assert_called_once_with(220, 680, 350, 720, duration=800)
    bot._input.tap.assert_called_once_with(1184, 550)
    assert result == (1184, 550)


def test_forbidden_attack_button_never_tapped_in_combat():
    bot = BotController.__new__(BotController)
    bot._loader = MagicMock()
    # Configuration with attack_button at (1450, 770) which is in forbidden feature dock zone
    bot._loader.load_ui_coords.return_value = {
        "combat": {
            "auto_button": {"x": 1308, "y": 241},
            "attack_button": {"x": 1450, "y": 770},
            "horse_button": {"x": 1553, "y": 370},
        }
    }
    bot._input = MagicMock()
    bot._screen = MagicMock()
    dummy_frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._screen.capture.return_value = dummy_frame
    bot._ensure_quest_panel_closed = MagicMock()
    bot._ensure_combat_skills_visible = MagicMock()
    bot._ensure_dismounted_from_horse = MagicMock()
    bot._collapse_chat_bar = MagicMock()
    bot._get_current_location_from_hud = MagicMock(return_value=("Dương Trung động", (100, 100)))
    bot._is_same_combat_map = MagicMock(return_value=True)
    bot._checkpoint = MagicMock()

    # Monster detected at safe ground (800, 500)
    cycle = [0]
    def mock_find(frame, monster, blocked_positions=None):
        cycle[0] += 1
        if cycle[0] == 1:
            return (800, 500)
        return None
    bot._find_monster_on_screen = MagicMock(side_effect=mock_find)

    # After cycle 1, quest finishes
    read_c = [0]
    def mock_read(frame, roi=None):
        read_c[0] += 1
        if read_c[0] >= 10:
            return [TextBox("Tiêu diệt Bát phụ Hà Đông Sư (10/10)", 0.95, 100, 300)]
        return []
    bot._text = MagicMock()
    bot._text.read.side_effect = mock_read
    bot._return_to_nearest_city_commerce_chief = MagicMock(return_value=True)

    ctx = StateContext(account_id="ACC_02")
    ctx.current_quest_type = QuestType.COMBAT
    ctx.extra = {
        "combat_map": "Dương Trung động",
        "combat_monster": "Bát phụ Hà Đông Sư",
        "combat_count": 10,
    }

    bot._handle_navigate(ctx)

    # Right-click called on monster
    bot._input.right_click.assert_called_with(800, 500, count=1)
    # BUT attack_btn (1450, 770) MUST NEVER have been tapped!
    for call in bot._input.tap.call_args_list:
        coords = call[0]
        assert coords != (1450, 770), "attack_btn at forbidden (1450, 770) was mistakenly tapped!"


def test_combat_source_has_no_auto_activation_path():
    import inspect

    source = inspect.getsource(BotController._handle_navigate)
    assert "Kích hoạt tự đánh" not in source
