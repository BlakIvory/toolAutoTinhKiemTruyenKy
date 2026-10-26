import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.bot_controller import BotController, BotState, StateContext
from src.text_recognition import TextBox

def test_navigate_ignores_interact_bubble_when_moving():
    bot = BotController.__new__(BotController)
    bot._checkpoint = MagicMock()
    bot._input = MagicMock()
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    bot._dismiss_unexpected_popups = MagicMock(return_value=False)
    bot._ensure_npc_dialog_closed = MagicMock()
    bot._press_buff_speed = MagicMock()
    bot._find_dialog_buttons = MagicMock(return_value=[])
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    bot._is_npc_interact_bubble_present = MagicMock(return_value=True)
    bot._click_npc_interact_bubble = MagicMock()
    bot._navigate_to_dialogue_npc = MagicMock(return_value=True)
    bot._text = MagicMock()
    bot._text.read.return_value = []

    # Giả lập tọa độ liên tục thay đổi (nhân vật đang chạy qua phố)
    coord_sequence = [(100, 100), (102, 105), (105, 110), (108, 115), (112, 120)]
    idx = 0
    def mock_hud(frame=None):
        nonlocal idx
        c = coord_sequence[min(idx, len(coord_sequence) - 1)]
        idx += 1
        return "Thành Đô", c

    bot._get_current_location_from_hud = MagicMock(side_effect=mock_hud)

    ctx = StateContext(account_id="ACC_01")
    ctx.extra = {
        "quest_target_city": "Thành Đô",
        "quest_target_npc": "Thái thú",
    }

    # Giới hạn số vòng để test nhanh
    with patch("time.sleep", return_value=None):
        # Chạy _handle_navigate nhưng giới hạn loop
        with patch.object(bot, "_switch_to_city", return_value=False):
            res = bot._handle_navigate(ctx)

    # Trong suốt lúc đang chạy, không được bấm bong bóng của NPC ven đường
    assert bot._click_npc_interact_bubble.call_count == 0
    # Vì hết giờ mà chưa mở đối thoại, bot phải điều hướng lại thay vì nhảy sang INTERACT_NPC
    assert res == BotState.NAVIGATE
    bot._navigate_to_dialogue_npc.assert_called_with(ctx, "Thành Đô", "Thái thú")

def test_navigate_clicks_interact_bubble_when_stationary():
    bot = BotController.__new__(BotController)
    bot._checkpoint = MagicMock()
    bot._input = MagicMock()
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    bot._dismiss_unexpected_popups = MagicMock(return_value=False)
    bot._ensure_npc_dialog_closed = MagicMock()
    bot._press_buff_speed = MagicMock()
    bot._is_npc_interact_bubble_present = MagicMock(return_value=True)
    bot._click_npc_interact_bubble = MagicMock()
    bot._text = MagicMock()
    bot._text.read.return_value = []

    # Giả lập nhân vật đã dừng lại tại điểm đến: tọa độ không đổi (stationary_count >= 2)
    bot._get_current_location_from_hud = MagicMock(return_value=("Thành Đô", (200, 200)))

    # Ban đầu chưa mở đối thoại, sau khi click bubble thì đối thoại mở
    dialog_open = False
    def mock_is_open(frame=None):
        return dialog_open
    bot._is_npc_dialog_open = MagicMock(side_effect=mock_is_open)
    bot._find_dialog_buttons = MagicMock(return_value=[])

    def on_click_bubble(frame=None):
        nonlocal dialog_open
        dialog_open = True
    bot._click_npc_interact_bubble.side_effect = on_click_bubble

    ctx = StateContext(account_id="ACC_01")
    ctx.extra = {
        "quest_target_city": "Thành Đô",
        "quest_target_npc": "Thái thú",
    }

    # Cho time.time() tăng dần để stable_for >= 0.8s
    times = [100.0, 100.5, 101.5, 102.0, 102.5, 103.0]
    t_idx = 0
    def mock_time():
        nonlocal t_idx
        val = times[min(t_idx, len(times) - 1)]
        t_idx += 1
        return val

    with patch("time.sleep", return_value=None), patch("time.time", side_effect=mock_time):
        res = bot._handle_navigate(ctx)

    assert bot._click_npc_interact_bubble.call_count >= 1
    assert res == BotState.INTERACT_NPC

def test_navigate_clicks_interact_bubble_when_target_npc_detected():
    bot = BotController.__new__(BotController)
    bot._checkpoint = MagicMock()
    bot._input = MagicMock()
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    bot._dismiss_unexpected_popups = MagicMock(return_value=False)
    bot._ensure_npc_dialog_closed = MagicMock()
    bot._press_buff_speed = MagicMock()
    bot._is_npc_interact_bubble_present = MagicMock(return_value=True)
    bot._click_npc_interact_bubble = MagicMock()

    # Thấy nhãn NPC "Thái thú" trên màn hình
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox("Thái thú", 0.95, 600, 300),
    ]

    # Dù tọa độ đang thay đổi, nhưng thấy đúng tên NPC đích thì cho phép tương tác
    coord_sequence = [(100, 100), (101, 101), (102, 102)]
    idx = 0
    def mock_hud(frame=None):
        nonlocal idx
        c = coord_sequence[min(idx, len(coord_sequence) - 1)]
        idx += 1
        return "Thành Đô", c
    bot._get_current_location_from_hud = MagicMock(side_effect=mock_hud)

    dialog_open = False
    bot._is_npc_dialog_open = MagicMock(side_effect=lambda *args: dialog_open)
    bot._find_dialog_buttons = MagicMock(return_value=[])

    def on_click_bubble(frame=None):
        nonlocal dialog_open
        dialog_open = True
    bot._click_npc_interact_bubble.side_effect = on_click_bubble

    ctx = StateContext(account_id="ACC_01")
    ctx.extra = {
        "quest_target_city": "Thành Đô",
        "quest_target_npc": "Thái thú",
    }

    with patch("time.sleep", return_value=None):
        res = bot._handle_navigate(ctx)

    assert bot._click_npc_interact_bubble.call_count >= 1
    assert res == BotState.INTERACT_NPC

def test_interact_npc_closes_wrong_xa_phu_dialog_and_renavigates():
    bot = BotController.__new__(BotController)
    bot._checkpoint = MagicMock()
    bot._input = MagicMock()
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    bot._get_current_location_from_hud = MagicMock(return_value=("Thành Đô", (200, 200)))
    bot._is_npc_dialog_open = MagicMock(return_value=True)
    bot._find_dialog_buttons = MagicMock(return_value=[(300, 500)])
    bot._ensure_npc_dialog_closed = MagicMock()
    bot._navigate_to_dialogue_npc = MagicMock(return_value=True)

    # Đang mở đối thoại Xa Phu với các từ khóa đặc trưng (Lệ phí 3 lượng)
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox("Xa Phu", 0.95, 500, 380),
        TextBox("Lệ phí 3 lượng", 0.95, 500, 420),
        TextBox("Rời khỏi", 0.95, 400, 600),
    ]

    ctx = StateContext(account_id="ACC_01")
    ctx.extra = {
        "quest_target_city": "Thành Đô",
        "quest_target_npc": "Thái thú",
    }

    with patch("time.sleep", return_value=None):
        res = bot._handle_interact_npc(ctx)

    # Phải đóng hộp thoại Xa Phu nhầm và điều hướng lại tới Thái thú
    bot._ensure_npc_dialog_closed.assert_called()
    bot._navigate_to_dialogue_npc.assert_called_with(ctx, "Thành Đô", "Thái thú")
    assert res == BotState.NAVIGATE


def test_navigate_to_dialogue_npc_switches_city_via_xa_phu_when_different_city():
    bot = BotController.__new__(BotController)
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {"navigation": {}}
    bot._input = MagicMock()
    bot._adb = MagicMock()
    bot._type_unicode_text = MagicMock()
    bot._click_map_search_result = MagicMock(return_value=True)
    bot._is_map_open = MagicMock(return_value=True)
    bot._ensure_map_open = MagicMock(return_value=True)
    bot._search_and_select_npc = MagicMock(return_value=True)
    bot._close_map = MagicMock()
    bot._press_buff_speed = MagicMock()

    # Ban đầu ở Dương Châu, sau khi switch_to_city thì ở Thành Đô
    hud_cities = ["Dương Châu", "Thành Đô", "Thành Đô"]
    h_idx = 0
    def mock_hud(frame=None):
        nonlocal h_idx
        c = hud_cities[min(h_idx, len(hud_cities) - 1)]
        h_idx += 1
        return c, (100, 100)

    bot._get_current_location_from_hud = MagicMock(side_effect=mock_hud)
    bot._switch_to_city = MagicMock(return_value=True)

    ctx = StateContext(account_id="ACC_04")
    with patch("time.sleep", return_value=None):
        res = bot._navigate_to_dialogue_npc(ctx, "Thành Đô", "Thái thú")

    assert res is True
    # Bắt buộc phải gọi _switch_to_city qua Xa Phu để dịch chuyển nhanh, không chạy bộ
    bot._switch_to_city.assert_called_once_with(ctx, "Thành Đô", force=False)


def test_navigate_to_dialogue_npc_skips_switch_city_when_already_in_target_city():
    bot = BotController.__new__(BotController)
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {"navigation": {}}
    bot._input = MagicMock()
    bot._adb = MagicMock()
    bot._type_unicode_text = MagicMock()
    bot._click_map_search_result = MagicMock(return_value=True)
    bot._is_map_open = MagicMock(return_value=True)
    bot._ensure_map_open = MagicMock(return_value=True)
    bot._search_and_select_npc = MagicMock(return_value=True)
    bot._close_map = MagicMock()
    bot._press_buff_speed = MagicMock()

    # Đang ở sẵn Thành Đô
    bot._get_current_location_from_hud = MagicMock(return_value=("Thành Đô", (200, 200)))
    bot._switch_to_city = MagicMock(return_value=True)

    ctx = StateContext(account_id="ACC_04")
    with patch("time.sleep", return_value=None):
        res = bot._navigate_to_dialogue_npc(ctx, "Thành Đô", "Thái thú")

    assert res is True
    # Không được gọi _switch_to_city nếu đã ở sẵn thành phố đích
    bot._switch_to_city.assert_not_called()


def test_navigate_clicks_npc_directly_when_bubble_absent():
    bot = BotController.__new__(BotController)
    bot._checkpoint = MagicMock()
    bot._input = MagicMock()
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._dismiss_unexpected_popups = MagicMock(return_value=False)
    bot._ensure_npc_dialog_closed = MagicMock()
    bot._press_buff_speed = MagicMock()
    # Bubble KHÔNG được phát hiện
    bot._is_npc_interact_bubble_present = MagicMock(return_value=False)
    bot._click_npc_interact_bubble = MagicMock()

    # Thấy tên "Xa phu" trên màn hình
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox("Xa phu", 0.95, 750, 400),
    ]

    # Đang ở Biện Kinh, dừng yên tại chỗ
    bot._get_current_location_from_hud = MagicMock(return_value=("Biện Kinh", (150, 150)))

    dialog_open = False
    bot._is_npc_dialog_open = MagicMock(side_effect=lambda *args: dialog_open)
    bot._find_dialog_buttons = MagicMock(return_value=[])
    bot._find_finish_dialogue_button = MagicMock(return_value=None)

    def on_tap(x, y):
        nonlocal dialog_open
        # Khi bot tap vào vị trí NPC (750, 435) -> mở hội thoại
        if x == 750 and y == 435:
            dialog_open = True
    bot._input.tap.side_effect = on_tap

    ctx = StateContext(account_id="ACC_02")
    ctx.extra = {
        "quest_target_city": "Biện Kinh",
        "quest_target_npc": "Xa phu",
    }

    with patch("time.sleep", return_value=None):
        res = bot._handle_navigate(ctx)

    # Đã click trực tiếp vào vị trí NPC dù bubble_present là False
    bot._input.tap.assert_called_with(750, 435)
    assert res == BotState.INTERACT_NPC


def test_navigate_stationary_via_frame_difference_when_coords_none():
    bot = BotController.__new__(BotController)
    bot._checkpoint = MagicMock()
    bot._input = MagicMock()
    bot._screen = MagicMock()
    # Hai frame giống hệt nhau -> frame difference = 0 -> stationary
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._dismiss_unexpected_popups = MagicMock(return_value=False)
    bot._ensure_npc_dialog_closed = MagicMock()
    bot._press_buff_speed = MagicMock()
    bot._is_npc_interact_bubble_present = MagicMock(return_value=False)

    # Không thấy tên NPC nhưng đã dừng yên -> fallback bubble tap
    bot._text = MagicMock()
    bot._text.read.return_value = []

    # HUD trả về thành phố nhưng tọa độ là None!
    bot._get_current_location_from_hud = MagicMock(return_value=("Biện Kinh", None))

    dialog_open = False
    bot._is_npc_dialog_open = MagicMock(side_effect=lambda *args: dialog_open)
    bot._find_dialog_buttons = MagicMock(return_value=[])
    bot._find_finish_dialogue_button = MagicMock(return_value=None)

    def mock_click_bubble(*args, **kwargs):
        nonlocal dialog_open
        dialog_open = True
        return True
    bot._click_npc_interact_bubble = MagicMock(side_effect=mock_click_bubble)

    ctx = StateContext(account_id="ACC_02")
    ctx.extra = {
        "quest_target_city": "Biện Kinh",
        "quest_target_npc": "Xa phu",
    }

    times = [100.0, 100.5, 101.5, 102.0, 102.5, 103.0]
    t_idx = 0
    def mock_time():
        nonlocal t_idx
        val = times[min(t_idx, len(times) - 1)]
        t_idx += 1
        return val

    with patch("time.sleep", return_value=None), patch("time.time", side_effect=mock_time):
        res = bot._handle_navigate(ctx)

    assert bot._click_npc_interact_bubble.call_count >= 1
    assert res == BotState.INTERACT_NPC


def test_dialogue_recognizes_hang_da_den():
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox("Xa phu", 0.95, 400, 300),
        TextBox("Hàng đã đến!", 0.95, 360, 600),
    ]
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)

    # Kiểm tra _is_npc_dialog_open nhận diện "Hàng đã đến!"
    assert bot._is_npc_dialog_open(frame) is True

    # Kiểm tra _find_dialogue_response_button tìm được nút (360, 600)
    pos = bot._find_dialogue_response_button(frame)
    assert pos == (360, 600)


