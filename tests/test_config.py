"""
Tests: Config Loader
"""

import pytest
from pathlib import Path
import yaml
import tempfile
import os

from src.config_loader import ConfigLoader, AppConfig


@pytest.fixture
def temp_config_dir(tmp_path):
    """Tạo thư mục config tạm với các file test."""
    # app.yaml
    app_data = {
        "emulator": {
            "type": "ldplayer9",
            "ldplayer_path": "C:/Program Files/ldplayer9box",
            "adb_host": "127.0.0.1",
            "adb_port": 5555,
            "width": 1280,
            "height": 720,
            "dpi": 240,
        },
        "adb": {"adb_path": "", "connect_timeout": 10, "command_timeout": 30},
        "game": {"package_name": "vn.com.titanstudio.tinhkiem"},
    }
    (tmp_path / "app.yaml").write_text(yaml.dump(app_data), encoding="utf-8")

    # accounts.yaml
    accounts_data = {
        "accounts": [
            {"id": "ACC_01", "username": "user1", "password": "pass1", "priority": 1},
            {"id": "ACC_02", "username": "user2", "password": "pass2", "priority": 2, "enabled": False},
        ],
        "run_order": "priority",
    }
    (tmp_path / "accounts.yaml").write_text(yaml.dump(accounts_data), encoding="utf-8")

    # templates.yaml
    templates_data = {"templates": {}}
    (tmp_path / "templates.yaml").write_text(yaml.dump(templates_data), encoding="utf-8")

    # maps.yaml
    maps_data = {"maps": [], "navigation_graph": {}}
    (tmp_path / "maps.yaml").write_text(yaml.dump(maps_data), encoding="utf-8")

    # npcs.yaml
    (tmp_path / "npcs.yaml").write_text(yaml.dump({"npcs": []}), encoding="utf-8")

    # shops.yaml
    (tmp_path / "shops.yaml").write_text(yaml.dump({"shops": []}), encoding="utf-8")

    # ui_coordinates.yaml
    coords_data = {
        "regions": {
            "full_screen": {"x1": 0, "y1": 0, "x2": 1280, "y2": 720},
            "bottom_right": {"x1": 900, "y1": 550, "x2": 1280, "y2": 720},
        }
    }
    (tmp_path / "ui_coordinates.yaml").write_text(yaml.dump(coords_data), encoding="utf-8")

    return tmp_path


def test_load_app_config(temp_config_dir):
    loader = ConfigLoader(temp_config_dir)
    cfg = loader.load_app()
    assert isinstance(cfg, AppConfig)
    assert cfg.emulator.width == 1280
    assert cfg.emulator.height == 720
    assert cfg.game.package_name == "vn.com.titanstudio.tinhkiem"


def test_load_accounts(temp_config_dir):
    loader = ConfigLoader(temp_config_dir)
    cfg = loader.load_accounts()
    assert len(cfg.accounts) == 2
    assert cfg.accounts[0].id == "ACC_01"
    assert cfg.accounts[0].enabled is True
    assert cfg.accounts[1].enabled is False


def test_get_enabled_accounts(temp_config_dir):
    loader = ConfigLoader(temp_config_dir)
    enabled = loader.get_enabled_accounts()
    assert len(enabled) == 1
    assert enabled[0].id == "ACC_01"


def test_get_region(temp_config_dir):
    loader = ConfigLoader(temp_config_dir)
    region = loader.get_region("full_screen")
    assert region == (0, 0, 1280, 720)


def test_missing_config_file(temp_config_dir):
    from src.error_codes import BotError, ErrorCode
    loader = ConfigLoader(temp_config_dir)
    (temp_config_dir / "app.yaml").unlink()
    with pytest.raises(BotError) as exc_info:
        loader._app = None
        loader.load_app()
    assert exc_info.value.code == ErrorCode.CFG_001


def test_invalid_yaml(temp_config_dir):
    from src.error_codes import BotError, ErrorCode
    (temp_config_dir / "app.yaml").write_text("{ invalid: yaml: :", encoding="utf-8")
    loader = ConfigLoader(temp_config_dir)
    with pytest.raises(BotError) as exc_info:
        loader.load_app()
    assert exc_info.value.code == ErrorCode.CFG_002
