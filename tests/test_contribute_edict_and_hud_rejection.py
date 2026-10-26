import pytest
from unittest.mock import MagicMock
from src.state_machine import BotState, StateContext, QuestType
from src.text_recognition import TextBox
from src.bot_controller import BotController
from src.quest_classifier import QuestClassifier
from src.config_loader import ConfigLoader


def test_classifier_rejects_tan_thu_and_world_quests():
    cfg = ConfigLoader().load_app()
    qc = QuestClassifier(None, cfg)
    # Background text containing Tân Thủ should be rejected as UNKNOWN
    res = qc.classify(text="Nhiem vu huong dan Tan Thu Gap Pham Trong Yem")
    assert res == QuestType.UNKNOWN

    res2 = qc.classify(text="Nhiem vu The gioi: Danh quai Thien Mon Tran")
    assert res2 == QuestType.UNKNOWN

    # Real Thương Hội quest should still classify
    res3 = qc.classify(text="Nhiem vu Thuong hoi: Den Bien Kinh gap Pham Trong Yem")
    assert res3 == QuestType.NPC


def test_meet_npc_clicks_contribute_when_edict_open():
    bot = BotController.__new__(BotController)
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    bot._is_npc_dialog_open = MagicMock(return_value=True)
    bot._open_commerce_chief_dialog = MagicMock(return_value=True)
    bot._sync_quest_progress_from_text = MagicMock()
    bot._save_progress = MagicMock()
    bot._input = MagicMock()
    bot._screen = MagicMock()
    bot._screen.capture.return_value = "dummy"
    bot._text = MagicMock()

    # Screen contains Chủ Thương Hội initial edict dialog
    bot._text.read.return_value = [
        TextBox(text="Hoang thuong vua ban chieu: Nhan si nao giup suc...", confidence=0.9, center_x=714, center_y=365),
        TextBox(text="Vi trieu dinh cong hien suc lyc", confidence=0.9, center_x=321, center_y=579),
        TextBox(text="Xem tinh hinh nhiem vu thuong hoi da hoan thanh", confidence=0.9, center_x=807, center_y=568),
        TextBox(text="Gioi thieu ti mi nhiem vu thuong hoi", confidence=0.9, center_x=1184, center_y=578),
        TextBox(text="Khong co hung thu lam nhiem vu nay", confidence=0.9, center_x=345, center_y=684),
    ]

    ctx = StateContext(account_id="ACC_01")
    ctx.extra = {"check_previous_quest_on_login": True}

    next_state = bot._handle_meet_npc(ctx)

    # Must transition to GET_QUEST
    assert next_state == BotState.GET_QUEST
    # Must tap "Vì triều đình cống hiến sức lực" at (321, 579)
    bot._input.tap.assert_called_with(321, 579)
    # check_previous_quest_on_login must be cleared
    assert "check_previous_quest_on_login" not in ctx.extra


def test_get_quest_does_not_classify_background_when_dialog_closed():
    bot = BotController.__new__(BotController)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = "dummy"
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    bot._check_active_commerce_quest = MagicMock(return_value=None)
    bot._classifier = MagicMock()
    bot._checkpoint = MagicMock()

    ctx = StateContext(account_id="ACC_01")
    next_state = bot._handle_get_quest(ctx)

    # Must return MEET_NPC to reopen dialog and tap 'Vì triều đình cống hiến sức lực'
    assert next_state == BotState.MEET_NPC
    # Classifier must NOT have been called with background HUD text
    bot._classifier.classify.assert_not_called()
