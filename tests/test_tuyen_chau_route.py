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
    bot._input.tap.assert_any_call(266, 222)
    # Verify Step 4: Types Tuyền Châu with telex
    bot._input_search_text.assert_any_call("Tuyền Châu", telex="tuyeefn%schaau", ctx=ctx)
    # Verify Step 5: Clicks Tuyen Chau map
    bot._click_map_search_result.assert_called_once_with("Tuyen Chau")
    # Verify Step 6: Taps find NPC tab
    bot._input.tap.assert_any_call(376, 161)
    # Verify Step 8: Types Chủ thương hội with telex
    bot._input_search_text.assert_any_call("Chủ thương hội", telex="chur%sthuwownwg%shooji", ctx=ctx)
    # Verify Step 9: Clicks Chu thuong hoi NPC
    bot._click_npc_search_result.assert_called_once_with("Chu thuong hoi")
    # Verify Close map & Speed buff
    bot._close_map.assert_called_once()
    bot._press_buff_speed.assert_called_once()

