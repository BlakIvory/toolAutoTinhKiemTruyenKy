from pathlib import Path
import yaml
import re
import pytest

UI_COORDS_FILE = Path("config/ui_coordinates.yaml")

def test_ui_coordinates_structure_and_merchant_spawns():
    content = UI_COORDS_FILE.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    assert "commerce_quest" in data
    assert "western_merchant_spawns" in data["commerce_quest"]
    spawns = data["commerce_quest"]["western_merchant_spawns"]
    assert len(spawns) >= 6
    for s in spawns:
        assert "city" in s
        assert "x" in s and isinstance(s["x"], int)
        assert "y" in s and isinstance(s["y"], int)


def test_merchant_spawns_regex_save_and_restore():
    original_content = UI_COORDS_FILE.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(original_content)
        spawns = data["commerce_quest"]["western_merchant_spawns"]
        
        # Test modifying first spawn
        test_spawns = [dict(s) for s in spawns]
        test_spawns[0]["x"] = 999
        test_spawns[0]["y"] = 888

        # Generate block
        lines = ["  western_merchant_spawns:"]
        for s in test_spawns:
            lines.append(f'    - {{ city: "{s["city"]}", x: {s["x"]}, y: {s["y"]} }}')
        new_block = "\n".join(lines) + "\n"

        pattern = r'(?s)^  western_merchant_spawns:.*?(?=^  [a-zA-Z0-9_]+:|\Z)'
        assert re.search(pattern, original_content, flags=re.MULTILINE)
        new_content = re.sub(pattern, new_block, original_content, count=1, flags=re.MULTILINE)

        # Write and re-read
        UI_COORDS_FILE.write_text(new_content, encoding="utf-8")
        reloaded = yaml.safe_load(UI_COORDS_FILE.read_text(encoding="utf-8"))
        
        # Check that modified spawn is 999, 888
        assert reloaded["commerce_quest"]["western_merchant_spawns"][0]["x"] == 999
        assert reloaded["commerce_quest"]["western_merchant_spawns"][0]["y"] == 888
        # Check that other parts of yaml are still intact
        assert "login" in reloaded
        assert "shop" in reloaded
    finally:
        # Restore original
        UI_COORDS_FILE.write_text(original_content, encoding="utf-8")

    # Verify restored cleanly
    restored = yaml.safe_load(UI_COORDS_FILE.read_text(encoding="utf-8"))
    assert restored["commerce_quest"]["western_merchant_spawns"][0]["x"] != 999


CUSTOM_DICT_FILE = Path("config/custom_dictionary.yaml")


def test_custom_dictionary_yaml_structure():
    assert CUSTOM_DICT_FILE.exists()
    content = CUSTOM_DICT_FILE.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    assert "custom_entities" in data
    entities = data["custom_entities"]
    assert len(entities) > 0

    categories = set()
    names = set()
    for ent in entities:
        assert "category" in ent
        assert "canonical_name" in ent
        assert "aliases" in ent
        categories.add(ent["category"])
        names.add(ent["canonical_name"])

    assert "NPC" in categories
    assert "Quái vật" in categories
    assert "Chu Bân Cử" in names


def test_load_custom_dictionary_into_classifier():
    from src.quest_classifier import (
        load_custom_dictionary,
        NPC_ALIASES,
        KNOWN_DIALOGUE_NPCS,
        KNOWN_MONSTER_ALIASES,
        CUSTOM_ITEM_ALIASES,
    )

    count = load_custom_dictionary(CUSTOM_DICT_FILE)
    assert count > 0

    # Test Chu Bân Cử aliases
    assert "chu ban cu" in NPC_ALIASES
    assert NPC_ALIASES["chu ban cu"] == "Chu Bân Cử"
    assert "Chu Bân Cử" in KNOWN_DIALOGUE_NPCS

    # Test Monster aliases
    assert "coc doc" in KNOWN_MONSTER_ALIASES
    assert KNOWN_MONSTER_ALIASES["coc doc"] == "Cóc độc"

    # Test Item aliases
    assert "luc tru nguyen thach" in CUSTOM_ITEM_ALIASES
    assert CUSTOM_ITEM_ALIASES["luc tru nguyen thach"] == "Lục Trụ Nguyên Thạch"


def test_mousewheel_child_hierarchy_check():
    # Test widget hierarchy check string logic
    def is_child_of(widget_str: str, parent_str: str) -> bool:
        return widget_str == parent_str or widget_str.startswith(parent_str + ".")

    parent = ".!launcherapp.!notebook.!settingstab"
    child_canvas = ".!launcherapp.!notebook.!settingstab.!canvas"
    child_btn = ".!launcherapp.!notebook.!settingstab.!canvas.!frame.!button"
    other_tab = ".!launcherapp.!notebook.!accountstab.!frame"
    other_subframe = ".!launcherapp.!notebook.!settingstab2.!btn"

    assert is_child_of(parent, parent) is True
    assert is_child_of(child_canvas, parent) is True
    assert is_child_of(child_btn, parent) is True
    assert is_child_of(other_tab, parent) is False
    assert is_child_of(other_subframe, parent) is False


def test_settings_tab_refresh_data_does_not_crash():
    from unittest.mock import MagicMock
    from launcher import SettingsTab

    tab = SettingsTab.__new__(SettingsTab)
    tab.master = MagicMock()
    mock_progress_tab = MagicMock()
    mock_progress_tab._refresh_data = MagicMock()
    tab.master.winfo_children.return_value = [tab, mock_progress_tab]

    # Must have _refresh_data and not raise AttributeError
    tab._refresh_data()
    mock_progress_tab._refresh_data.assert_called_once()

