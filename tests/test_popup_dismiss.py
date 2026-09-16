from pathlib import Path
import yaml
from src.text_recognition import normalize_game_text


def test_popup_close_coordinates_defined():
    coords = yaml.safe_load(Path("config/ui_coordinates.yaml").read_text(encoding="utf-8"))
    nav = coords["navigation"]
    assert "settings_close_button" in nav
    assert nav["settings_close_button"] == {"x": 1412, "y": 170}
    assert "chat_close_button" in nav
    assert nav["chat_close_button"] == {"x": 795, "y": 105}
    assert "arena_close_button" in nav
    assert nav["arena_close_button"] == {"x": 1475, "y": 120}


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
