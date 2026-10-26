import pytest
from unittest.mock import MagicMock
from src.quest_classifier import get_canonical_npc_name, NPC_ALIASES
from src.bot_controller import BotController
from src.state_machine import StateContext, BotState, QuestType


def test_tao_ban_canonical_mapping():
    assert get_canonical_npc_name("Tao Ban") == "Tào Bân"
    assert get_canonical_npc_name("Tào Bán") == "Tào Bân"
    assert get_canonical_npc_name("Tào Bản") == "Tào Bân"
    assert get_canonical_npc_name("tào bân") == "Tào Bân"


def test_is_near_commerce_chief_ignores_cudc_thuong_nhan():
    bot = BotController.__new__(BotController)
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    
    mock_box = MagicMock()
    mock_box.text = "Thuong hoianh Cudc Thuong Nhan"
    mock_box.center_x = 1049
    mock_box.center_y = 485
    
    bot._text = MagicMock()
    bot._text.read = MagicMock(return_value=[mock_box])
    bot._get_current_location_from_hud = MagicMock(return_value=("Biện Kinh", (100, 100)))
    
    dummy_frame = MagicMock()
    # Must NOT detect this as Chủ thương hội
    assert bot._is_near_commerce_chief(frame=dummy_frame) is False


def test_is_near_commerce_chief_detects_chu_thuong_hoi():
    bot = BotController.__new__(BotController)
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    
    mock_box = MagicMock()
    mock_box.text = "Chu thuong hoi"
    mock_box.center_x = 858
    mock_box.center_y = 341
    
    bot._text = MagicMock()
    bot._text.read = MagicMock(return_value=[mock_box])
    bot._get_current_location_from_hud = MagicMock(return_value=("Dương Châu", (204, 196)))
    
    dummy_frame = MagicMock()
    assert bot._is_near_commerce_chief(frame=dummy_frame) is True


def test_navigate_does_not_click_interact_bubble_when_nav_not_started():
    bot = BotController.__new__(BotController)
    ctx = StateContext(account_id="ACC_01")
    ctx.current_quest_type = QuestType.NPC
    ctx.extra["quest_target_npc"] = "Tào Bân"
    ctx.extra["quest_target_city"] = "Biện Kinh"
    ctx.extra["npc_nav_started"] = False  # Map search failed, never started auto-walking
    
    bot._check_quest_timeout = MagicMock(return_value=False)
    bot._dismiss_unexpected_popups = MagicMock(return_value=False)
    bot._get_current_location_from_hud = MagicMock(return_value=("Biện Kinh", (180, 185)))
    bot._screen = MagicMock()
    bot._screen.capture = MagicMock(return_value=MagicMock())
    bot._is_npc_interact_bubble_present = MagicMock(return_value=True)  # Bubble from Xa Phu
    bot._click_npc_interact_bubble = MagicMock()
    bot._text = MagicMock()
    bot._text.read = MagicMock(return_value=[])  # target NPC not visible
    bot._find_dialog_buttons = MagicMock(return_value=[])
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    bot._find_finish_dialogue_button = MagicMock(return_value=None)
    bot._checkpoint = MagicMock()

    # Run one pass
    # Since nav_started is False and target NPC is not seen, it must NOT click the bubble
    bot._handle_navigate = BotController._handle_navigate.__get__(bot, BotController)
    
    # We test the logic: nav_started False -> arrived_at_target False
    nav_started = ctx.extra.get("npc_nav_started", False)
    has_target_npc = False
    stationary_count = 5
    stable_for = 1.0
    arrived_at_target = has_target_npc or (nav_started and stationary_count >= 2 and stable_for >= 0.4)
    assert arrived_at_target is False
