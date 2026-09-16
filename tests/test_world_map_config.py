from pathlib import Path

import yaml


def test_world_map_button_matches_1600x900_layout():
    config = yaml.safe_load(Path("config/ui_coordinates.yaml").read_text(encoding="utf-8"))
    assert config["navigation"]["map_world_button"] == {"x": 1436, "y": 782}
