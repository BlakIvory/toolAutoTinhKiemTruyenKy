import inspect

from src.bot_controller import BotController


def test_stale_target_hud_expires_and_combat_keeps_hunting():
    source = inspect.getsource(BotController._handle_navigate)

    assert "is_targeting_monster and (time.time() - last_target_seen) < 4.0" in source
    assert "total_combat_elapsed >= 600.0" not in source
    assert "time.time() - last_obstruction_check_time >= 10.0" in source


def test_completion_banner_low_on_screen_returns_to_turn_in_quest():
    source = inspect.getsource(BotController._handle_navigate)

    assert "(200, 380, 1400, 800)" in source
    assert 'banner_text = " ".join' in source
    assert '"nhiem vu thuong hoi"' in source
    assert "Quay về Chủ thương hội trả nhiệm vụ ngay" in source
    assert "return BotState.RETURN_QUEST" in source


def test_combat_never_opens_quest_panel_or_clicks_horse_button():
    source = inspect.getsource(BotController._handle_navigate)
    dismount_source = inspect.getsource(BotController._ensure_dismounted_from_horse)

    assert "self._check_quest_panel(ctx)" not in source
    assert "self._input.tap(*h_pos)" not in dismount_source
