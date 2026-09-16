from pathlib import Path

import yaml


def test_western_merchant_spawn_coordinates_are_exact():
    config = yaml.safe_load(Path("config/ui_coordinates.yaml").read_text(encoding="utf-8"))
    actual = {
        (spawn["city"], spawn["x"], spawn["y"])
        for spawn in config["commerce_quest"]["western_merchant_spawns"]
    }
    assert actual == {
        ("Tuyền Châu", 187, 191),
        ("Biện Kinh", 179, 184),
        ("Thành Đô", 224, 225),
        ("Tương Dương", 187, 180),
        ("Dương Châu", 206, 194),
        ("Đại Lý", 178, 187),
    }
