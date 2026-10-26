from unittest.mock import MagicMock
from src.bot_controller import BotController
from src.state_machine import StateContext, QuestType

def test_press_buff_speed_disabled_in_combat():
    bot = BotController.__new__(BotController)
    loader = MagicMock()
    loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "buff_button": {"x": 1308, "y": 160}
        }
    }
    bot._loader = loader
    bot._input = MagicMock()

    # Case 1: ctx with COMBAT quest type
    ctx_combat = StateContext(account_id="acc1")
    ctx_combat.current_quest_type = QuestType.COMBAT

    bot._press_buff_speed(ctx=ctx_combat)
    assert bot._input.tap.call_count == 0

    # Case 2: _in_combat flag is True
    bot._in_combat = True
    bot._press_buff_speed()
    assert bot._input.tap.call_count == 0

    # Case 3: Not in combat -> should tap normally
    bot._in_combat = False
    ctx_dialogue = StateContext(account_id="acc1")
    ctx_dialogue.current_quest_type = QuestType.NPC
    bot._press_buff_speed(ctx=ctx_dialogue)
    assert bot._input.tap.call_count == 1
    bot._input.tap.assert_called_with(1308, 160)
