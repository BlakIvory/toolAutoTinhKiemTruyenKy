import time
from unittest.mock import MagicMock, patch
import numpy as np
from src.bot_controller import BotController
from src.state_machine import BotState, StateContext
from src.text_recognition import TextBox
from src.quest_classifier import QuestType


def test_full_shop_quest_cycle_to_completion():
    """
    Test toàn bộ chu trình hoàn chỉnh từ nhận nhiệm vụ mua đồ đến hoàn thành 1/20:
    1. Nhận NV tại Chủ thương hội
    2. Phân loại NV (Loại 2 - Shop)
    3. Di chuyển liên thành (đo tốc độ, phát hiện di chuyển)
    4. Tìm và mua đồ từ Thương nhân Tây Vực (chọn đúng món, bấm mua)
    5. Quay về Chủ thương hội và bấm 'Nhiệm vụ đã hoàn thành!'
    6. Ghi nhận thành công quest_done (1/20)
    """
    bot = BotController.__new__(BotController)
    bot._app_cfg = MagicMock()
    bot._app_cfg.mode.dry_run = False
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._text = MagicMock()
    bot._vision = MagicMock()
    bot._input = MagicMock()
    bot._adb = MagicMock()
    bot._checkpoint = MagicMock()
    bot._dismiss_unexpected_popups = MagicMock(return_value=True)
    bot._ensure_npc_dialog_closed = MagicMock(return_value=True)
    bot._ensure_map_open = MagicMock(return_value=True)
    bot._is_map_open = MagicMock(return_value=False)
    bot._close_map = MagicMock(return_value=True)
    bot._press_buff_speed = MagicMock(return_value=True)
    bot._input_search_text = MagicMock(return_value=True)
    bot._click_map_search_result = MagicMock(return_value=True)
    bot._click_npc_search_result = MagicMock(return_value=True)
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    bot._open_commerce_chief_dialog = MagicMock(return_value=True)
    # Successful quest completion automatically closes the NPC dialog.
    bot._is_npc_dialog_open = MagicMock(return_value=False)

    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "navigation": {
            "map_tab_find_map": {"x": 176, "y": 162},
            "map_tab_find_npc": {"x": 376, "y": 161},
            "map_search_input": {"x": 266, "y": 222},
            "map_tuyen_chau": {"x": 115, "y": 315},
        },
        "commerce_quest": {
            "complete_quest_fallback": {"x": 361, "y": 693},
            "western_merchant_spawns": [
                {"city": "Tương Dương", "x": 187, "y": 191}
            ]
        },
            "shop": {
                "dialog_buy_option": {"x": 272, "y": 566},
                "western_merchant_items": {
                    "Linh Dược Tây Vực": {"x": 397, "y": 190, "row": 1, "col": 2}
                },
                "grid_slots": [
                {"x": 316, "y": 190},
                {"x": 397, "y": 190}
            ],
            "buy_button": {"x": 538, "y": 485},
            "confirm_button": {"x": 465, "y": 786},
            "close_button": {"x": 702, "y": 128},
            "inventory_close": {"x": 1577, "y": 81},
            "tooltip_title_region": {"x1": 700, "y1": 30, "x2": 1100, "y2": 150}
        }
    }

    ctx = StateContext(account_id="ACC_02")
    ctx.total_quests_done = 0

    # Bước 1: Hội thoại nhận nhiệm vụ mới
    dialog_contrib = TextBox("Vì triều đình cống hiến sức lực", 0.95, 360, 560)
    bot._text.read.return_value = [dialog_contrib]
    bot._check_active_commerce_quest = MagicMock(return_value=("Mua Linh dược Tây Vực cho Thương hội", (100, 200)))
    
    # Bước 2: Phân loại nhiệm vụ (Loại 2 - Shop)
    ctx.extra["quest_text"] = "Mua Linh dược Tây Vực cho Thương hội"
    bot._go_to_city_coordinate = MagicMock(return_value=True)
    
    merchant_box = TextBox("Thương nhân Tây Vực", 0.95, 750, 420)
    dong_y_mua = TextBox("Đồng ý mua", 0.95, 272, 566)
    shop_header = TextBox("Cửa hiệu Tạp hóa", 0.95, 400, 100)
    item_tooltip = TextBox("Linh dược Tây Vực", 0.95, 850, 60)
    complete_btn = TextBox("Nhiệm vụ đã hoàn thành!", 0.95, 361, 693)
    reward_msg = TextBox("Chúc mừng ngươi đã hoàn thành!", 0.95, 800, 450)

    def mock_read(frame, roi=None):
        # Quantity popup
        if roi == (250, 520, 1350, 900):
            return [TextBox("Số lượng mua", 0.95, 500, 600), TextBox("Đồng ý", 0.95, 465, 786)]
        # NPC Dialog options
        if roi and roi[1] >= 250:
            if ctx.extra.get("returning_quest"):
                return [complete_btn]
            return [dialog_contrib]
        # Tooltip inspection in shop
        if roi == (700, 30, 1100, 150):
            return [item_tooltip]
        # General screen reads
        if ctx.extra.get("returning_quest"):
            return [reward_msg]
        return [merchant_box, dong_y_mua, shop_header]

    bot._text.read.side_effect = mock_read
    bot._text.best_match.return_value = (0.95, merchant_box, "Thương nhân Tây Vực")
    bot._get_current_location_from_hud = MagicMock(return_value=("Tương Dương", (187, 191)))

    # Bước 3: Mua đồ
    ctx.extra["requested_shop_item"] = "Linh dược Tây Vực"
    next_state = bot._handle_buy_item(ctx)
    assert next_state == BotState.RETURN_QUEST
    assert ctx.extra.get("returning_quest") is True

    # Bước 4: Trả nhiệm vụ khi returning_quest == True
    bot._get_current_location_from_hud = MagicMock(return_value=("Tuyền Châu", (192, 183)))
    bot._handle_meet_npc(ctx)

    # Đảm bảo hoàn thành thành công 1 nhiệm vụ
    assert ctx.total_quests_done == 1
    assert ctx.extra.get("returning_quest") is None


def test_complete_quest_clears_all_classification_state():
    bot = BotController.__new__(BotController)
    bot._db = MagicMock()
    bot._checkpoint = MagicMock()
    bot._get_username = MagicMock(return_value="tester")

    ctx = StateContext(account_id="ACC_CLEANUP")
    ctx.current_quest_type = QuestType.SPECIAL_NPC_SHOP
    stale_keys = (
        "quest_text", "yellow_tokens", "yellow_item", "requested_shop_item",
        "combat_map", "combat_monster", "combat_count", "combat_sect",
        "quest_target_city", "quest_target_npc", "quest_detail_text",
        "returning_quest", "chief_nav_started", "teleport_to_combat_quest",
        "auto_walking", "merchant_search_retries",
    )
    ctx.extra.update({key: "stale" for key in stale_keys})

    with patch("src.bot_controller.time.sleep", return_value=None), patch(
        "src.bot_controller.activity.quest_done", return_value=None
    ):
        next_state = bot._handle_complete_quest(ctx)

    assert next_state == BotState.CHECK_LOCATION
    assert ctx.current_quest_type == QuestType.UNKNOWN
    assert all(key not in ctx.extra for key in stale_keys)


def test_missing_western_merchant_keeps_searching_instead_of_returning_to_chief():
    bot = BotController.__new__(BotController)
    bot._classifier = MagicMock()
    bot._classifier.classify.return_value = QuestType.SPECIAL_NPC_SHOP
    bot._checkpoint = MagicMock()
    bot._save_progress = MagicMock()
    bot._find_western_merchant = MagicMock(return_value=False)

    ctx = StateContext(account_id="ACC_03")
    ctx.extra.update({
        "quest_text": "Mua Linh Dược Tây Vực ở Thương nhân Tây Vực",
        "requested_shop_item": "Linh Dược Tây Vực",
        "merchant_search_retries": 7,
    })

    with patch("src.bot_controller.time.sleep", return_value=None):
        next_state = bot._handle_classify_quest(ctx)

    assert next_state == BotState.CLASSIFY_QUEST
    assert ctx.extra["merchant_search_retries"] == 8
    assert ctx.extra["requested_shop_item"] == "Linh Dược Tây Vực"
    bot._find_western_merchant.assert_called_once_with(ctx)
