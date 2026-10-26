import sys
sys.path.insert(0, ".")
import os
import cv2
import numpy as np
from unittest.mock import MagicMock
from src.bot_controller import BotController
from src.state_machine import BotState, StateContext
from src.text_recognition import TextBox, GameTextRecognizer


def test_is_npc_dialog_open_on_user_screenshot():
    user_img_path = r"C:\Users\nguye\.gemini\antigravity-ide\brain\febb5fe8-ac33-4940-8181-6d712ca9f599\.user_uploaded\media_1790156221177.png"
    if not os.path.exists(user_img_path):
        return

    img = cv2.imread(user_img_path)
    img_1080 = cv2.resize(img, (1920, 1080))

    bot = BotController.__new__(BotController)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = img_1080
    bot._text = GameTextRecognizer()
    bot._vision = MagicMock()
    bot._vision.find_template.return_value = MagicMock(found=False)

    is_open = BotController._is_npc_dialog_open(bot, img_1080)
    assert is_open is True, "Hộp thoại Chủ thương hội trên ảnh screenshot phải được nhận diện là đang mở!"


def test_handle_meet_npc_clicks_complete_box():
    bot = BotController.__new__(BotController)
    bot._open_commerce_chief_dialog = MagicMock(return_value=True)
    bot._check_active_commerce_quest = MagicMock(return_value=None)
    # Game closes the NPC dialog automatically after a successful turn-in.
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bot._input = MagicMock()
    bot._checkpoint = MagicMock()
    bot._ensure_npc_dialog_closed = MagicMock()
    bot._dismiss_unexpected_popups = MagicMock(return_value=True)

    # Simulated OCR boxes matching the actual game screenshot
    prompt_box = TextBox(
        "Nhiem vu lan truoc da hoan thanh chua? Nhola trongngay phai hoan thanh 20 nhiem vy nhe!",
        0.95, 837, 437
    )
    # OCR often merges 'vu da' into 'vudahoan'
    complete_box = TextBox("Nhiem vudahoan thanh!", 0.95, 361, 693)
    status_box = TextBox("Xem tinh hinh nhiem vu thuong hoi da hoan th", 0.95, 958, 681)
    not_done_box = TextBox("Nhiem vu van chua hoan thanh", 0.95, 1337, 693)

    reward_box = TextBox("Chuc mung nguoi da hoan thanh!", 0.95, 800, 450)
    bot._text = MagicMock()
    bot._text.read.side_effect = [
        [prompt_box, complete_box, status_box, not_done_box], # Dialog read
        [reward_box], # Response read after clicking complete
        []
    ]

    ctx = StateContext(account_id="ACC_01")
    ctx.extra = {"returning_quest": True}

    bot._handle_meet_npc(ctx)

    # Must click on the complete button at (361, 693)
    tap_calls = bot._input.tap.call_args_list
    assert any(c[0] == (361, 693) for c in tap_calls), f"Bot did not click complete button! Calls: {tap_calls}"
    assert ctx.total_quests_done == 1


def test_handle_meet_npc_syncs_from_progress_summary_dialog():
    bot = BotController.__new__(BotController)
    bot._open_commerce_chief_dialog = MagicMock(return_value=True)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bot._input = MagicMock()
    bot._checkpoint = MagicMock()
    bot._ensure_npc_dialog_closed = MagicMock()
    bot._save_progress = MagicMock()
    bot._get_username = MagicMock(return_value="test_user")

    # Game summary dialog: "Hôm nay đã hoàn thành 3 lần nhiệm vụ. Còn có thể nhận 17 lần!" with button "Kết thúc đối thoại"
    summary_box = TextBox("Hom nay da hoan thanh 3 lan nhiemvy.Conco thenhan 171an!", 0.95, 450, 400)
    close_box = TextBox("Ket thuc doi thoai", 0.95, 300, 600)

    bot._text = MagicMock()
    bot._text.read.return_value = [summary_box, close_box]

    ctx = StateContext(account_id="ACC_04")
    ctx.total_quests_done = 0

    next_state = bot._handle_meet_npc(ctx)

    assert ctx.total_quests_done == 3
    assert ctx.extra.get("in_game_progress_synced") is True
    assert next_state == BotState.MEET_NPC
    tap_calls = bot._input.tap.call_args_list
    assert any(c[0] == (300, 600) for c in tap_calls)


def test_handle_meet_npc_recognizes_no_active_quest_and_clicks_contribute():
    """Kiểm tra: khi hộp thoại có 'Hoàng thượng vừa ban chiếu' và nút 'Vì triều đình cống hiến sức lực',
    bot phải nhận diện là CHƯA CÓ NHIỆM VỤ NÀO, bấm nút nhận nhiệm vụ và chuyển sang GET_QUEST,
    tuyệt đối không bấm nút Hoàn thành hoặc báo chưa hoàn thành nhiệm vụ."""
    bot = BotController.__new__(BotController)
    bot._open_commerce_chief_dialog = MagicMock(return_value=True)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bot._input = MagicMock()
    bot._checkpoint = MagicMock()
    bot._save_progress = MagicMock()

    banner_box = TextBox(
        "Hoang thuong vua ban chieu: Nhan si nao giup suc Trieu dinh se duoc trong thuong! Moi ngay co the nhan toi da 20 lan nhiem vu!",
        0.95, 750, 340
    )
    contrib_btn = TextBox("Vi trieu dinh cong hien suc luc", 0.95, 330, 575)
    view_status_btn = TextBox("Xem tinh hinh nhiem vu thuong hoi da hoan thanh", 0.95, 800, 575)

    bot._text = MagicMock()
    bot._text.read.return_value = [banner_box, contrib_btn, view_status_btn]

    ctx = StateContext(account_id="ACC_02")
    # Giả sử cờ returning_quest hoặc check_previous_quest_on_login còn sót lại
    ctx.extra = {
        "returning_quest": True,
        "check_previous_quest_on_login": True,
    }

    next_state = bot._handle_meet_npc(ctx)

    assert next_state == BotState.GET_QUEST
    assert "returning_quest" not in ctx.extra
    assert "check_previous_quest_on_login" not in ctx.extra
    tap_calls = bot._input.tap.call_args_list
    assert any(c[0] == (330, 575) for c in tap_calls), f"Bot did not click contribute button! Calls: {tap_calls}"


