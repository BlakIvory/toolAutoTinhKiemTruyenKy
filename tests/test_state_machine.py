"""
Tests: State Machine
"""

import pytest
from src.state_machine import StateMachine, BotState, StateContext, QuestType


@pytest.fixture
def mock_app_config():
    from src.config_loader import (
        AppConfig, EmulatorConfig, ADBConfig, GameConfig
    )
    return AppConfig(
        emulator=EmulatorConfig(),
        adb=ADBConfig(),
        game=GameConfig(package_name="vn.com.test"),
    )


@pytest.fixture
def sm(mock_app_config):
    return StateMachine(mock_app_config)


def test_state_transition(sm):
    """Test chuyển state cơ bản."""
    transitions = []

    def handler_boot(ctx: StateContext) -> BotState:
        return BotState.CONNECT_EMULATOR

    def handler_connect(ctx: StateContext) -> BotState:
        return BotState.FINISHED

    def on_change(old, new, ctx):
        transitions.append((old, new))

    sm.register(BotState.BOOT, handler_boot)
    sm.register(BotState.CONNECT_EMULATOR, handler_connect)
    sm.set_state_change_callback(on_change)

    sm.run("TEST_ACCOUNT", initial_state=BotState.BOOT)

    assert (BotState.BOOT, BotState.CONNECT_EMULATOR) in transitions
    assert (BotState.CONNECT_EMULATOR, BotState.FINISHED) in transitions
    assert sm.context.current_state == BotState.FINISHED


def test_error_handling(sm):
    """Test BotError → ERROR state."""
    from src.error_codes import BotError, ErrorCode

    def handler_boot(ctx: StateContext) -> BotState:
        raise BotError(ErrorCode.ADB_001, "Test error")

    sm.register(BotState.BOOT, handler_boot)
    sm.run("TEST_ACCOUNT", initial_state=BotState.BOOT)

    assert sm.context.current_state == BotState.ERROR
    assert sm.context.error_code == ErrorCode.ADB_001


def test_context_retry(sm):
    """Test retry counter trong context."""
    call_count = [0]

    def handler_boot(ctx: StateContext) -> BotState:
        call_count[0] += 1
        if ctx.retry_count < 2:
            ctx.retry_count += 1
            return BotState.BOOT  # Retry
        return BotState.FINISHED

    sm.register(BotState.BOOT, handler_boot)
    sm.run("TEST_ACCOUNT", initial_state=BotState.BOOT)

    assert call_count[0] == 3
    assert sm.context.current_state == BotState.FINISHED


def test_quest_type_enum():
    """Test QuestType enum values."""
    assert QuestType.NPC == "NPC"
    assert QuestType.UNKNOWN == "UNKNOWN"
    assert QuestType.SPECIAL_NPC_SHOP == "SPECIAL_NPC_SHOP"


def test_bot_state_enum():
    """Test BotState enum."""
    assert BotState.BOOT == "BOOT"
    assert BotState.FINISHED == "FINISHED"
    assert BotState.ERROR == "ERROR"
