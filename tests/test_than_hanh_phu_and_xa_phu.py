import pytest
from unittest.mock import MagicMock, call, patch
from src.bot_controller import BotController
from src.state_machine import StateContext, BotState
from src.text_recognition import TextBox


def test_interact_bubble_never_uses_fixed_fallback_when_not_detected():
    import numpy as np

    bot = BotController.__new__(BotController)
    bot._vision = MagicMock()
    bot._vision.find_template.return_value = MagicMock(found=False)
    bot._input = MagicMock()
    bot._loader = MagicMock()
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)

    assert bot._click_npc_interact_bubble(frame) is False
    bot._input.tap.assert_not_called()


def test_xa_phu_close_range_taps_are_limited_to_two():
    bot = BotController.__new__(BotController)
    bot._get_current_location_from_hud = MagicMock(return_value=("Bien Kinh", (170, 182)))
    bot._is_bag_open = MagicMock(return_value=False)
    bot._ensure_map_open = MagicMock(return_value=True)
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "navigation": {"map_tab_find_npc": {"x": 376, "y": 161}}
    }
    bot._input = MagicMock()
    bot._click_npc_search_result = MagicMock(return_value=True)
    bot._is_map_open = MagicMock(return_value=False)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = MagicMock()
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox(text="Xa phu", confidence=0.95, center_x=800, center_y=400)
    ]
    bot._is_npc_interact_bubble_present = MagicMock(return_value=True)
    bot._click_npc_interact_bubble = MagicMock(return_value=True)
    bot._walk_intercity_via_world_map = MagicMock(return_value=True)

    clock = iter(range(100))
    with patch("src.bot_controller.time.time", side_effect=lambda: next(clock)), \
         patch("src.bot_controller.time.sleep"):
        result = bot._switch_to_city(StateContext(account_id="ACC_01"), "Tuyen Chau")

    assert result is True
    # Never tap the NPC label/body because the game can treat it as a new
    # ground-movement command. Only the real interaction bubble is allowed.
    assert bot._input.tap.call_args_list.count(((800, 435),)) == 0
    assert bot._click_npc_interact_bubble.call_count == 2


def test_tuyen_chau_fixed_xa_phu_coordinate_clicks_only_chat_bubble_once():
    bot = BotController.__new__(BotController)
    bot._get_current_location_from_hud = MagicMock(return_value=("Tuyền Châu", (188, 179)))
    bot._is_bag_open = MagicMock(return_value=False)
    bot._ensure_map_open = MagicMock(return_value=True)
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "navigation": {"map_tab_find_npc": {"x": 376, "y": 161}},
        "xa_phu_arrival": {
            "Tuyền Châu": {"x": 188, "y": 179, "tolerance": 3, "max_interact_clicks": 1}
        },
    }
    bot._input = MagicMock()
    bot._search_tuyen_chau_xa_phu = MagicMock(return_value=True)
    bot._is_map_open = MagicMock(return_value=False)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = MagicMock()
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox(text="Xa phu Tuyền Châu", confidence=0.95, center_x=800, center_y=400)
    ]
    bot._is_npc_interact_bubble_present = MagicMock(return_value=True)
    location_calls_when_clicked = []
    bot._click_npc_interact_bubble = MagicMock(
        side_effect=lambda *_: location_calls_when_clicked.append(
            bot._get_current_location_from_hud.call_count
        ) or True
    )
    bot._dismiss_unexpected_popups = MagicMock(return_value=False)
    bot._walk_intercity_via_world_map = MagicMock(return_value=True)

    clock = iter(range(100))
    with patch("src.bot_controller.time.time", side_effect=lambda: next(clock)), \
         patch("src.bot_controller.time.sleep"):
        result = bot._switch_to_city(StateContext(account_id="ACC_01"), "Dương Châu")

    assert result is True
    bot._click_npc_interact_bubble.assert_called_once()
    # 1 lần đọc HUD ban đầu + 1 lần ở frame đến đích: phải click ngay,
    # không chờ frame thứ hai trong khi auto-path tiếp tục chạy.
    assert location_calls_when_clicked == [2]
    # Sau lệnh auto-path ban đầu, Tuyền Châu chỉ được bấm
    # bong bóng hội thoại; không phát thêm tap/swipe để nhích nhân vật.
    assert bot._input.tap.call_args_list == [call(376, 161)]
    bot._input.swipe.assert_not_called()


def test_travel_confirm_rejects_player_name_containing_di():
    boxes = [
        TextBox(text="BaoTriDienLanh", confidence=0.95, center_x=607, center_y=529),
        TextBox(text="SHOP VIP", confidence=0.95, center_x=760, center_y=580),
    ]

    assert BotController._find_travel_confirm_button(boxes) is None


def test_city_name_alone_does_not_mean_xa_phu_dialog_is_open():
    boxes = [
        TextBox(text="Thành Đô", confidence=0.95, center_x=500, center_y=400),
        TextBox(text="Tuyền Châu", confidence=0.95, center_x=1450, center_y=40),
    ]
    assert BotController._has_xa_phu_dialog_evidence(boxes) is False


def test_overhead_npc_and_chat_city_text_do_not_falsely_trigger_xa_phu_dialog():
    # Trường hợp nhân vật vừa tới Xa phu Tuyền Châu: trên màn hình có nhãn NPC Xa phu và chat Đến: Tuyền Châu
    boxes = [
        TextBox(text="Xa phu Tuyền Châu", confidence=0.95, center_x=500, center_y=450),
        TextBox(text="Đến: Tuyền Châu(188, 180)", confidence=0.95, center_x=450, center_y=720),
        TextBox(text="Thương Thắng Đô", confidence=0.95, center_x=600, center_y=500),
    ]
    assert BotController._has_xa_phu_dialog_evidence(boxes, cur_city="Tuyền Châu") is False



def test_quest_tracker_text_does_not_mark_open_xa_phu_picker_as_wrong_npc_dialog():
    boxes = [
        # HUD/quest tracker text can mention another NPC while Xa Phu is open.
        TextBox(text="Vo cong", confidence=0.95, center_x=170, center_y=430),
        TextBox(text="Thanh Do (3 luong)", confidence=0.95, center_x=480, center_y=560),
    ]
    assert BotController._has_other_npc_dialog_evidence(boxes) is False
    assert BotController._has_xa_phu_dialog_evidence(boxes) is True


def test_other_npc_dialog_detection_uses_central_dialog_text():
    boxes = [TextBox(text="Vo cong", confidence=0.95, center_x=720, center_y=540)]
    assert BotController._has_other_npc_dialog_evidence(boxes) is True


@pytest.mark.parametrize(
    "text", ["Lệ phí 3 lượng", "Muốn đi đâu", "Không đi đâu cả"]
)
def test_xa_phu_dialog_requires_distinctive_text(text):
    boxes = [TextBox(text=text, confidence=0.95, center_x=500, center_y=560)]
    assert BotController._has_xa_phu_dialog_evidence(boxes) is True


@pytest.mark.parametrize("label", ["Đi", "Đồng ý", "Xác nhận", "Chấp nhận"])
def test_travel_confirm_accepts_only_real_button_labels(label):
    expected = TextBox(text=label, confidence=0.95, center_x=720, center_y=560)

    assert BotController._find_travel_confirm_button([expected]) is expected

def test_xa_phu_direct_navigation():
    bot = BotController.__new__(BotController)
    bot._get_current_location_from_hud = MagicMock(side_effect=[
        ("Biện Kinh", (170, 182)),
        ("Biện Kinh", (170, 182)),
        ("Tuyền Châu", (192, 182)),
    ])
    bot._is_bag_open = MagicMock(return_value=False)
    bot._ensure_map_open = MagicMock(return_value=True)
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "navigation": {
            "map_tab_find_npc": {"x": 376, "y": 161},
            "map_search_input": {"x": 266, "y": 222}
        },
        "commerce_quest": {
            "xa_phu_cities": {"Tuyền Châu": {"x": 719, "y": 561}}
        }
    }
    bot._input = MagicMock()
    bot._click_npc_search_result = MagicMock(return_value=True)
    bot._is_map_open = MagicMock(return_value=False)
    bot._close_map = MagicMock()
    bot._screen = MagicMock()
    bot._screen.capture.return_value = MagicMock()
    bot._is_npc_interact_bubble_present = MagicMock(return_value=False)
    bot._is_npc_dialog_open = MagicMock(return_value=True)
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox(text="Tuyền Châu", confidence=0.9, center_x=720, center_y=560),
        TextBox(text="Lệ phí 3 lượng", confidence=0.9, center_x=500, center_y=500),
    ]
    bot._check_xa_phu_out_of_money = MagicMock(return_value=False)

    ctx = StateContext(account_id="ACC_01")
    res = bot._switch_to_city(ctx, "Tuyền Châu")
    
    assert res is True

def test_xa_phu_auto_close_bag_if_open():
    bot = BotController.__new__(BotController)
    bot._get_current_location_from_hud = MagicMock(side_effect=[
        ("Biện Kinh", (170, 182)),
        ("Biện Kinh", (170, 182)),
        ("Tuyền Châu", (192, 182)),
    ])
    bot._is_bag_open = MagicMock(return_value=True)
    bot._close_bag = MagicMock()
    bot._ensure_map_open = MagicMock(return_value=True)
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "navigation": {
            "map_tab_find_npc": {"x": 376, "y": 161},
            "map_search_input": {"x": 266, "y": 222}
        },
        "commerce_quest": {
            "xa_phu_cities": {"Tuyền Châu": {"x": 719, "y": 561}}
        }
    }
    bot._input = MagicMock()
    bot._click_npc_search_result = MagicMock(return_value=True)
    bot._is_map_open = MagicMock(return_value=False)
    bot._close_map = MagicMock()
    bot._screen = MagicMock()
    bot._screen.capture.return_value = MagicMock()
    bot._is_npc_interact_bubble_present = MagicMock(return_value=False)
    bot._is_npc_dialog_open = MagicMock(return_value=True)
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox(text="Tuyền Châu", confidence=0.9, center_x=720, center_y=560),
        TextBox(text="Lệ phí 3 lượng", confidence=0.9, center_x=500, center_y=500),
    ]
    bot._check_xa_phu_out_of_money = MagicMock(return_value=False)

    ctx = StateContext(account_id="ACC_01")
    res = bot._switch_to_city(ctx, "Tuyền Châu")
    
    assert res is True
    bot._close_bag.assert_called_once()

def test_xa_phu_out_of_money_logging():
    bot = BotController.__new__(BotController)
    bot._get_current_location_from_hud = MagicMock(return_value=("Biện Kinh", (170, 182)))
    bot._is_bag_open = MagicMock(return_value=False)
    bot._ensure_map_open = MagicMock(return_value=True)
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "navigation": {
            "map_tab_find_npc": {"x": 376, "y": 161},
            "map_search_input": {"x": 266, "y": 222}
        }
    }
    bot._input = MagicMock()
    bot._click_npc_search_result = MagicMock(return_value=True)
    bot._is_map_open = MagicMock(return_value=False)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = MagicMock()
    bot._is_npc_interact_bubble_present = MagicMock(return_value=False)
    bot._is_npc_dialog_open = MagicMock(return_value=True)
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox(text="Tuyền Châu", confidence=0.9, center_x=720, center_y=560),
        TextBox(text="Lệ phí 3 lượng", confidence=0.9, center_x=500, center_y=500),
    ]
    bot._check_xa_phu_out_of_money = MagicMock(return_value=True)
    bot._checkpoint = MagicMock()
    bot._ensure_npc_dialog_closed = MagicMock()
    bot._walk_intercity_via_world_map = MagicMock(return_value=True)

    ctx = StateContext(account_id="ACC_01")
    res = bot._switch_to_city(ctx, "Tuyền Châu")
    
    assert res is True
    assert ctx.extra.get("out_of_money") is True
    bot._walk_intercity_via_world_map.assert_called_once_with(ctx, "Tuyền Châu", max_wait=120.0, retry_local_xa_phu=False)


def test_check_xa_phu_out_of_money_detects_co_tien_thi_hay_ngoi_xe():
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox(text="Có tiền thì hãy ngồi xe!", confidence=0.95, center_x=500, center_y=450),
    ]
    frame = object()
    assert bot._check_xa_phu_out_of_money(frame) is True


def test_has_xa_phu_dialog_evidence_recognizes_co_tien_thi_hay_ngoi_xe():
    boxes = [
        TextBox(text="Có tiền thì hãy ngồi xe!", confidence=0.95, center_x=500, center_y=450),
    ]
    assert BotController._has_xa_phu_dialog_evidence(boxes, cur_city="Tuyền Châu") is True


def test_out_of_money_cleared_on_complete_quest():
    from src.state_machine import BotState
    bot = BotController.__new__(BotController)
    bot._db = MagicMock()
    bot._checkpoint = MagicMock()
    bot._get_username = MagicMock(return_value="test_user")

    ctx = StateContext(account_id="ACC_01")
    ctx.extra["out_of_money"] = True

    with patch("src.bot_controller.time.sleep"):
        next_state = bot._handle_complete_quest(ctx)

    assert next_state == BotState.CHECK_LOCATION
    assert "out_of_money" not in ctx.extra

