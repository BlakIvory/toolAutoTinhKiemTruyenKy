from pathlib import Path
import yaml
from src.text_recognition import normalize_game_text


def test_popup_close_coordinates_defined():
    coords = yaml.safe_load(Path("config/ui_coordinates.yaml").read_text(encoding="utf-8"))
    nav = coords["navigation"]
    assert "settings_close_button" in nav
    assert nav["settings_close_button"] == {"x": 1412, "y": 170}
    assert "chat_close_button" in nav
    assert nav["chat_close_button"] == {"x": 774, "y": 135}
    assert "chat_collapse_button" in nav
    assert nav["chat_collapse_button"] == {"x": 490, "y": 843}
    assert "arena_close_button" in nav
    assert nav["arena_close_button"] == {"x": 1475, "y": 120}


def test_collapse_chat_bar_never_taps_bottom_chat():
    from unittest.mock import MagicMock
    import numpy as np
    from src.bot_controller import BotController
    from src.text_recognition import TextBox

    bot = BotController.__new__(BotController)
    loader = MagicMock()
    loader.load_ui_coords.return_value = {
        "navigation": {
            "chat_collapse_button": {"x": 490, "y": 843}
        }
    }
    bot._loader = loader
    bot._input = MagicMock()
    bot._screen = MagicMock()
    bot._text = MagicMock()
    bot._is_map_open = MagicMock(return_value=False)

    dummy_frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._screen.capture.return_value = dummy_frame

    # Expanded chat bar with "tong hop" tab
    bot._text.read.return_value = [TextBox("tong hop", 0.99, 500, 800)]

    res = bot._collapse_chat_bar(dummy_frame)
    assert res is False
    bot._input.tap.assert_not_called()


def test_prepare_ingame_hud_after_login_collapses_quest_tracker_once(monkeypatch):
    from unittest.mock import MagicMock, call
    from src.bot_controller import BotController
    from src.state_machine import StateContext

    bot = BotController.__new__(BotController)
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "navigation": {
            "quest_tracker_collapse_button": {"x": 62, "y": 165},
            "chat_collapse_button": {"x": 490, "y": 843},
        }
    }
    bot._input = MagicMock()
    bot._checkpoint = MagicMock()
    monkeypatch.setattr("src.bot_controller.time.sleep", lambda _seconds: None)
    ctx = StateContext(account_id="ACC_01")

    bot._prepare_ingame_hud_after_login(ctx)
    bot._prepare_ingame_hud_after_login(ctx)

    assert bot._input.tap.call_args_list == [call(62, 165)]
    assert ctx.extra["login_hud_collapsed"] is True
    bot._checkpoint.save.assert_called_once_with(ctx)


def test_ensure_npc_dialog_closed_never_taps_unfinished_notice(monkeypatch):
    from unittest.mock import MagicMock
    import numpy as np
    from src.bot_controller import BotController
    from src.text_recognition import TextBox

    bot = BotController.__new__(BotController)
    bot._is_map_open = MagicMock(return_value=False)
    bot._is_npc_dialog_open = MagicMock(side_effect=[True, False])
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox("Nguoi chua hoan thanh nhiem vy ta giao! Tiep tyc co gang nhe!", 0.99, 456, 339)
    ]
    bot._input = MagicMock()
    monkeypatch.setattr("src.bot_controller.time.sleep", lambda _seconds: None)

    assert bot._ensure_npc_dialog_closed(max_attempts=1) is True
    bot._input.tap.assert_called_once_with(800, 200)


def test_map_coordinates_not_landing_in_chat():
    coords = yaml.safe_load(Path("config/ui_coordinates.yaml").read_text(encoding="utf-8"))
    cq = coords["commerce_quest"]
    # Chat channels are located below y=800 (y ~ 820..880).
    # Map coordinate input boxes and confirm button must be at y=783 on map panel.
    assert cq["map_coordinate_x"]["y"] <= 790
    assert cq["map_coordinate_y"]["y"] <= 790
    assert cq["map_coordinate_confirm"]["y"] <= 790


def test_popup_keyword_normalization():
    assert "tat nguoi choi" in normalize_game_text("Tắt người chơi")
    assert "kenh chat" in normalize_game_text("Kênh chat")
    assert "chien truong" in normalize_game_text("Chiến trường")
    assert "tu dong sap xep" in normalize_game_text("Bạn đồng ý dùng chế độ tự động sắp xếp không?")


def test_shop_close_and_inventory_close_coordinates():
    coords = yaml.safe_load(Path("config/ui_coordinates.yaml").read_text(encoding="utf-8"))
    shop = coords["shop"]
    # Close buttons must be at y ~ 110 (top diamond 'X' buttons), NOT hitting Sắp xếp (y=156)
    assert shop["close_button"] == {"x": 679, "y": 110}
    assert shop["inventory_close"] == {"x": 1535, "y": 109}
    assert shop["inventory_close"]["y"] < 130, "inventory_close must be < 130 to avoid clicking Sắp xếp button"
    assert shop["sort_cancel_button"] == {"x": 1075, "y": 600}


def test_shop_item_matching_distinctive_tokens():
    from src.bot_controller import BotController
    # Tinh Kim Tay Vuc should NOT match Linh Duoc Tay Vuc (even though both have 'Tay Vuc')
    match_wrong, score_wrong = BotController._is_matching_shop_item("LinhdugcTay Vy", "Tinh Kim Tay Vuc")
    assert not match_wrong, f"Linh Dược should not match Tinh Kim, got score={score_wrong}"

    # Tinh Kim Tay Vuc should match Tinh Kim
    match_ok, score_ok = BotController._is_matching_shop_item("Tinh Kim Tay Vuc", "Tinh Kim Tay Vuc")
    assert match_ok, f"Expected match for Tinh Kim, score={score_ok}"

    # Linh Duoc Tay Vuc matches LinhdugcTay Vy
    match_linh, score_linh = BotController._is_matching_shop_item("LinhdugcTay Vy", "Linh Duoc Tay Vuc")
    assert match_linh, f"Expected match for Linh Duoc, score={score_linh}"

    # Bao Tay matches BaotaolayVtc
    match_bao, score_bao = BotController._is_matching_shop_item("BaotaolayVtc", "Bao Tay Tay Vuc")
    assert match_bao, f"Expected match for Bao Tay, score={score_bao}"

    # Kim cang nguyen thach should NOT match Am nguyen thach
    match_am, score_am = BotController._is_matching_shop_item("Ám nguyên thạch", "Kim cang nguyên thạch")
    assert not match_am, f"Ám nguyên thạch should NOT match Kim cang nguyên thạch, got score={score_am}"
    assert score_am < 0.40, f"Score for conflicting attribute should be penalized (<0.40), got {score_am}"

    # Ám Yên and Âm/Ám Phách are distinct fixed shop slots; accent removal
    # must not let the wrong qualifier pass fuzzy matching.
    match_yen_to_phach, score_yen_to_phach = BotController._is_matching_shop_item(
        "Ám Yên Nguyên Thạch", "Âm Phách Nguyên Thạch"
    )
    assert not match_yen_to_phach, f"Ám Yên must not match Âm Phách (score={score_yen_to_phach})"

    match_phach_to_yen, score_phach_to_yen = BotController._is_matching_shop_item(
        "Âm Phách Nguyên Thạch", "Ám Yên Nguyên Thạch"
    )
    assert not match_phach_to_yen, f"Âm Phách must not match Ám Yên (score={score_phach_to_yen})"

    match_phach, score_phach = BotController._is_matching_shop_item(
        "Âm Phách Nguyên Thạch", "Ám Phách Nguyên Thạch"
    )
    assert match_phach, f"Accent-insensitive Âm/Ám Phách should match itself (score={score_phach})"

    # Kim cang nguyen thach matches Kim cang nguyen thach
    match_kim, score_kim = BotController._is_matching_shop_item("Kim cang nguyên thạch", "Kim cang nguyên thạch")
    assert match_kim, f"Expected match for Kim cang nguyên thạch, score={score_kim}"
    assert score_kim >= 0.90

    # Kim cang nguyen thach matches OCR variations: 'Kim Cang Nguyen Thac' and 'KimCangNguyenThac'
    match_kim_ocr1, score_kim_ocr1 = BotController._is_matching_shop_item("Kim Cang Nguyen Thac", "Kim cang nguyên thạch")
    assert match_kim_ocr1, f"Expected match for OCR 'Kim Cang Nguyen Thac', score={score_kim_ocr1}"
    assert score_kim_ocr1 >= 0.85

    match_kim_ocr2, score_kim_ocr2 = BotController._is_matching_shop_item("KimCangNguyenThac", "Kim cang nguyên thạch")
    assert match_kim_ocr2, f"Expected match for OCR 'KimCangNguyenThac', score={score_kim_ocr2}"
    assert score_kim_ocr2 >= 0.85

    # Ensure match_linh and match_bao also meet >= 0.80 threshold
    assert score_linh >= 0.80, f"Linh Duoc score should be >= 0.80, got {score_linh}"
    assert score_bao >= 0.80, f"Bao Tay score should be >= 0.80, got {score_bao}"


def test_dismiss_unexpected_popups_ignores_hud_icons():
    """HUD top-right buttons like 'Ngự các', 'Phúc lợi' must never trigger popup dismiss or tap minimap."""
    from unittest.mock import MagicMock
    import numpy as np
    from src.bot_controller import BotController
    from src.text_recognition import TextBox

    bot = BotController.__new__(BotController)
    loader = MagicMock()
    loader.load_ui_coords.return_value = {
        "navigation": {
            "settings_close_button": {"x": 1412, "y": 170}
        }
    }
    bot._loader = loader
    bot._input = MagicMock()
    bot._screen = MagicMock()
    bot._text = MagicMock()
    bot._is_map_open = MagicMock(return_value=False)

    dummy_frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._screen.capture.return_value = dummy_frame

    # Simulated HUD icon "Ngự các" at top-right bar (x=1150, y=140)
    bot._text.read.return_value = [
        TextBox("Ngu cac", 0.95, 1150, 140),
        TextBox("Phuc loi", 0.95, 930, 140),
        TextBox("Hang", 0.95, 1040, 140),
    ]

    res = bot._dismiss_unexpected_popups(dummy_frame)
    assert res is False, "Top HUD icons must not be treated as popups!"
    bot._input.tap.assert_not_called()


def test_warning_popup_without_button_uses_back_and_never_taps_game():
    from unittest.mock import MagicMock
    import numpy as np
    from src.bot_controller import BotController
    from src.text_recognition import TextBox

    bot = BotController.__new__(BotController)
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {"navigation": {}}
    bot._input = MagicMock()
    bot._adb = MagicMock()
    bot._screen = MagicMock()
    bot._text = MagicMock()
    bot._is_map_open = MagicMock(return_value=False)
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._text.read.return_value = [
        TextBox("Cảnh Báo", 0.99, 300, 205),
        TextBox("Hiện tại xuất hiện fanpage giả mạo NPH", 0.95, 650, 260),
        TextBox("Thường xuyên thay đổi mật khẩu", 0.95, 650, 390),
    ]

    assert bot._dismiss_unexpected_popups(frame) is True
    bot._input.press_back.assert_called_once()
    bot._input.tap.assert_not_called()


def test_switch_to_city_from_external_map_bypasses_xa_phu():
    """In external/cave maps like Tường Vân động, bot must directly walk intercity via world map."""
    from unittest.mock import MagicMock
    from src.bot_controller import BotController
    from src.state_machine import StateContext, BotState

    bot = BotController.__new__(BotController)
    ctx = StateContext(account_id="ACC_01", current_state=BotState.NAVIGATE)
    bot._is_bag_open = MagicMock(return_value=False)
    bot._get_current_location_from_hud = MagicMock(return_value=("Tuong", (207, 202)))
    bot._walk_intercity_via_world_map = MagicMock(return_value=True)

    res = bot._switch_to_city(ctx, "Tuyền Châu")
    assert res is True
    bot._walk_intercity_via_world_map.assert_called_once_with(ctx, "Tuyền Châu", max_wait=90.0, retry_local_xa_phu=True)


def test_dismiss_dialogue_completion_banner():
    from unittest.mock import MagicMock
    import numpy as np
    from src.bot_controller import BotController

    bot = BotController.__new__(BotController)
    bot._input = MagicMock()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    res = bot._dismiss_dialogue_completion_banner(frame)
    assert res is False
    # Tuyệt đối không bấm vào thanh chat (735, 808)
    assert bot._input.tap.call_count == 0



