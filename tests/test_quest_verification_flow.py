import pytest
from unittest.mock import MagicMock
import numpy as np
from src.bot_controller import BotController
from src.state_machine import StateContext, BotState, QuestType
from src.text_recognition import TextBox


def test_step1_check_completion_when_finished():
    """Kiểm tra bước 1: Tìm Chủ thương hội, nhấn hoàn thành nhiệm vụ, nếu xong thì nhận NV tiếp."""
    bot = BotController.__new__(BotController)
    bot._app_cfg = MagicMock()
    bot._app_cfg.mode.dry_run = False
    bot._input = MagicMock()
    bot._checkpoint = MagicMock()
    bot._db = MagicMock()
    bot._save_progress = MagicMock()
    bot._dismiss_unexpected_popups = MagicMock(return_value=True)
    bot._press_buff_speed = MagicMock()
    bot._get_username = MagicMock(return_value="test_user")

    # Khi bấm Hoàn thành, game đóng đối thoại (is_dialog_still_open == False)
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    bot._open_commerce_chief_dialog = MagicMock(return_value=True)
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    complete_box = TextBox("Nhiệm vụ đã hoàn thành!", 0.95, 360, 693)
    contribute_box = TextBox("Vì triều đình cống hiến sức lực", 0.95, 330, 575)

    bot._text = MagicMock()
    bot._text.read.side_effect = [
        [complete_box],       # Mở đối thoại có nút Hoàn thành
        [contribute_box],     # Mở lại đối thoại để nhận nhiệm vụ mới
    ]

    ctx = StateContext(account_id="ACC_TEST")
    ctx.extra["check_previous_quest_on_login"] = True

    next_state = bot._handle_meet_npc(ctx)

    # Đã bấm nút hoàn thành
    bot._input.tap.assert_any_call(360, 693)
    # Nhiệm vụ được tính hoàn thành thành công
    assert ctx.total_quests_done == 1
    # Mở lại đối thoại và bấm nhận NV mới
    bot._input.tap.assert_any_call(330, 575)
    assert next_state == BotState.GET_QUEST


def test_step1_check_completion_when_not_finished_routes_to_fetch_info():
    """Kiểm tra bước 1 khi chưa hoàn thành: đối thoại vẫn mở -> chuyển sang lấy thông tin NV."""
    bot = BotController.__new__(BotController)
    bot._app_cfg = MagicMock()
    bot._app_cfg.mode.dry_run = False
    bot._input = MagicMock()
    bot._checkpoint = MagicMock()
    bot._ensure_npc_dialog_closed = MagicMock(return_value=True)
    bot._press_buff_speed = MagicMock()
    bot._get_username = MagicMock(return_value="test_user")

    # Hộp thoại vẫn mở sau khi bấm hoàn thành
    bot._is_npc_dialog_open = MagicMock(return_value=True)
    bot._open_commerce_chief_dialog = MagicMock(return_value=True)
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    complete_box = TextBox("Nhiệm vụ đã hoàn thành!", 0.95, 360, 693)
    warning_box = TextBox("Ngươi chưa hoàn thành nhiệm vụ Thương hội, hãy mau đi làm!", 0.95, 800, 450)

    bot._text = MagicMock()
    bot._text.read.side_effect = [
        [complete_box],  # Hộp thoại ban đầu
        [warning_box],   # Sau khi bấm hoàn thành: NPC báo chưa xong
    ]
    bot._check_active_commerce_quest = MagicMock(return_value=(
        "Đến gặp Triệu Diên Niên tại Tương Dương", (100, 300)
    ))

    ctx = StateContext(account_id="ACC_TEST")
    ctx.extra["check_previous_quest_on_login"] = True

    next_state = bot._handle_meet_npc(ctx)

    # Đã bấm nút hoàn thành để kiểm tra
    bot._input.tap.assert_any_call(360, 693)
    # Không tăng số lượng hoàn thành
    assert ctx.total_quests_done == 0
    # Đã gọi _check_active_commerce_quest để lấy thông tin nhiệm vụ
    bot._check_active_commerce_quest.assert_called_once()
    assert ctx.extra["quest_text"] == "Đến gặp Triệu Diên Niên tại Tương Dương"
    assert next_state == BotState.CLASSIFY_QUEST


def test_step2_quest_panel_clicks_merchant_quest_directly():
    """Kiểm tra bước 2: Mở bảng NV -> Bỏ qua Tuần hoàn, Click thẳng Nhiệm vụ Thương hội -> Đọc chi tiết."""
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._screen = MagicMock()
    bot._input = MagicMock()
    bot._loader = MagicMock()
    bot._find_blue_npc_link = MagicMock(return_value=None)
    bot._loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "quest_button": {"x": 1460, "y": 765},
            "quest_panel_close": {"x": 1466, "y": 125},
            "quest_panel_merchant_row": {"x": 474, "y": 398},
        }
    }

    dummy_frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._screen.capture.return_value = dummy_frame

    # Header OCR: chưa mở -> mở -> Bảng nhiệm vụ
    bot._is_quest_panel_open = MagicMock(side_effect=[False, True, True, True, True])

    # Sequence of reads:
    # 1. Tìm icon Nhiệm vụ góc phải dưới
    # 2. Quét danh sách bên trái: tìm thấy 'Nhiệm vụ Thương hội'
    # 3. Đọc chi tiết bảng bên phải
    icon_nv = TextBox("Nhiệm vụ", 0.95, 1460, 765)
    merchant_box = TextBox("Nhiệm vụ Thương hội", 0.95, 474, 398)
    detail_box = TextBox("Đến Biện Kinh đối thoại với Triệu Diên Niên", 0.95, 900, 300)

    bot._text.read.side_effect = [
        [icon_nv],       # Đọc icon góc phải dưới
        [merchant_box],  # Quét danh sách: click thẳng vào 'Nhiệm vụ Thương hội'
        [detail_box],    # Đọc chi tiết bên phải
    ]

    ctx = StateContext(account_id="ACC_TEST")
    result = bot._check_quest_panel(ctx)

    assert result is not None
    quest_text, _ = result
    assert "Triệu Diên Niên" in quest_text

    # Xác nhận các bước click theo đúng thứ tự:
    tap_calls = [call.args for call in bot._input.tap.call_args_list]

    # 1. Click icon Nhiệm vụ góc phải dưới (1460, 765)
    assert (1460, 765) in tap_calls
    # 2. Click thẳng 'Nhiệm vụ Thương hội' (474, 398)
    assert (474, 398) in tap_calls
    # 3. Đóng bảng nhiệm vụ (1466, 125)
    assert (1466, 125) in tap_calls
    # 4. Tuyệt đối KHÔNG click tọa độ Tuần hoàn (473, 350)
    assert (473, 350) not in tap_calls
