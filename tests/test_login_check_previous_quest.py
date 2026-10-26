import pytest
from unittest.mock import MagicMock
import numpy as np
from src.bot_controller import BotController
from src.state_machine import StateContext, BotState, QuestType
from src.text_recognition import TextBox


def test_login_flow_checks_previous_quest_and_completes_it():
    """Vừa đăng nhập: đến Chủ thương hội, bấm hoàn thành, hộp thoại đóng -> mở lại nhận NV mới."""
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

    # Khi kiểm tra sau khi bấm Hoàn thành: hộp thoại đóng (is_dialog_still_open == False)
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    bot._open_commerce_chief_dialog = MagicMock(return_value=True)
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    prompt_box = TextBox("Nhiệm vụ lần trước đã hoàn thành chưa?", 0.95, 800, 400)
    complete_box = TextBox("Nhiệm vụ đã hoàn thành!", 0.95, 360, 693)
    teleport_box = TextBox("Đưa ta đến nơi làm nhiệm vụ", 0.95, 760, 575)
    contribute_box = TextBox("Vì triều đình cống hiến sức lực", 0.95, 330, 575)

    bot._text = MagicMock()
    bot._text.read.side_effect = [
        [prompt_box, complete_box, teleport_box],  # Phải ưu tiên trả NV, không được dịch chuyển
        [contribute_box],            # Sau khi mở lại đối thoại: hiện nút cống hiến
    ]

    ctx = StateContext(account_id="ACC_01")
    ctx.extra["check_previous_quest_on_login"] = True

    next_state = bot._handle_meet_npc(ctx)

    # Đã bấm nút hoàn thành
    bot._input.tap.assert_any_call(360, 693)
    assert not any(call.args == (760, 575) for call in bot._input.tap.call_args_list)
    # Nhiệm vụ được tính hoàn thành
    assert ctx.total_quests_done == 1
    assert "check_previous_quest_on_login" not in ctx.extra
    # Mở lại đối thoại thành công và bấm nhận NV mới
    bot._input.tap.assert_any_call(330, 575)
    assert next_state == BotState.GET_QUEST


def test_login_flow_unfinished_quest_continues_current_quest():
    """Vừa đăng nhập: đến Chủ thương hội, bấm hoàn thành nhưng chưa xong -> đóng đối thoại, tiếp tục làm NV."""
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

    prompt_box = TextBox("Nhiệm vụ lần trước đã hoàn thành chưa?", 0.95, 800, 400)
    complete_box = TextBox("Nhiệm vụ đã hoàn thành!", 0.95, 360, 693)
    warning_box = TextBox("Ngươi chưa hoàn thành nhiệm vụ Thương hội, hãy mau đi làm!", 0.95, 800, 450)

    bot._text = MagicMock()
    bot._text.read.side_effect = [
        [prompt_box, complete_box],  # Đọc hộp thoại lần 1
        [warning_box],               # Sau khi bấm hoàn thành: NPC báo chưa xong
    ]
    bot._check_active_commerce_quest = MagicMock(return_value=(
        "Tiêu diệt 10 Thực Nhân Thảo", (100, 300)
    ))

    ctx = StateContext(account_id="ACC_02")
    ctx.extra["check_previous_quest_on_login"] = True

    next_state = bot._handle_meet_npc(ctx)

    # Đã bấm nút hoàn thành để kiểm tra
    bot._input.tap.assert_any_call(360, 693)
    # Không tăng số lượng hoàn thành
    assert ctx.total_quests_done == 0
    assert "check_previous_quest_on_login" not in ctx.extra
    # Đã đóng hộp thoại và buff tốc chạy
    bot._ensure_npc_dialog_closed.assert_called_once()
    bot._press_buff_speed.assert_called_once()
    assert ctx.extra["quest_text"] == "Tiêu diệt 10 Thực Nhân Thảo"
    assert next_state == BotState.CLASSIFY_QUEST


def test_meet_npc_no_active_quest_clicks_contribute_button():
    """Khi chưa có nhiệm vụ: Chủ thương hội có nút 'Vì triều đình cống hiến sức lực' -> Nhấn để nhận NV."""
    bot = BotController.__new__(BotController)
    bot._app_cfg = MagicMock()
    bot._app_cfg.mode.dry_run = False
    bot._input = MagicMock()
    bot._checkpoint = MagicMock()
    bot._save_progress = MagicMock()
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    bot._open_commerce_chief_dialog = MagicMock(return_value=True)
    bot._is_npc_dialog_open = MagicMock(return_value=True)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    edict_box = TextBox(
        "Hoàng thượng vừa ban chiếu: Nhân sĩ nào giúp sức Triều đình sẽ được trọng thưởng! Mỗi ngày có thể nhận tối đa 20 lần nhiệm vụ!",
        0.95, 800, 350
    )
    contribute_box = TextBox("Vì triều đình cống hiến sức lực", 0.95, 330, 575)
    history_box = TextBox("Xem tình hình nhiệm vụ thương hội đã hoàn thành", 0.95, 600, 575)
    intro_box = TextBox("Giới thiệu tỉ mỉ nhiệm vụ thương hội", 0.95, 900, 575)
    reject_box = TextBox("Không có hứng thú làm nhiệm vụ này", 0.95, 330, 700)

    bot._text = MagicMock()
    bot._text.read.return_value = [edict_box, contribute_box, history_box, intro_box, reject_box]

    ctx = StateContext(account_id="ACC_03")
    ctx.extra["check_previous_quest_on_login"] = True

    next_state = bot._handle_meet_npc(ctx)

    # Đã bấm nút "Vì triều đình cống hiến sức lực" (330, 575)
    bot._input.tap.assert_called_once_with(330, 575)
    # Tuyệt đối không bấm nhầm nút lịch sử hay nút hoàn thành fallback
    assert (600, 575) not in [call.args for call in bot._input.tap.call_args_list]
    assert (360, 693) not in [call.args for call in bot._input.tap.call_args_list]
    # Cờ kiểm tra đăng nhập được xóa
    assert "check_previous_quest_on_login" not in ctx.extra
    # Chuyển sang GET_QUEST
    assert next_state == BotState.GET_QUEST

