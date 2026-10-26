from pathlib import Path
import pytest
import yaml

from src.bot_controller import BotController
from src.text_recognition import normalize_game_text


def test_western_merchant_items_configuration():
    coords = yaml.safe_load(Path("config/ui_coordinates.yaml").read_text(encoding="utf-8"))
    shop = coords["shop"]
    assert "western_merchant_items" in shop, "shop section must contain western_merchant_items"
    items = shop["western_merchant_items"]

    # Must contain all 17 items
    assert len(items) >= 17, f"Expected at least 17 items, got {len(items)}"

    expected_items = [
        "Bảo Đao Tây Vực",
        "Linh Dược Tây Vực",
        "Lục Trụ Nguyên Thạch",
        "Hộ Uyển Lông Lạc Đà",
        "Thịt Lạc Đà",
        "Lông Đại Bàng",
        "Tinh Kim Tây Vực",
        "Ám Yên Nguyên Thạch",
        "Pháo Hoa",
        "Trứng Đại Bàng",
        "Âm Phách Nguyên Thạch",
        "Dạ Quang Bôi",
        "Tuyết Tinh Nguyên Thạch",
        "Thanh Kim Nguyên Thạch",
        "Hổ Phách Nguyên Thạch",
        "Lông Lạc Đà",
        "Kim Cang Nguyên Thạch",
    ]

    for item_name in expected_items:
        assert item_name in items, f"Missing item: {item_name}"
        data = items[item_name]
        assert "x" in data and "y" in data, f"Missing x or y for {item_name}"
        assert "row" in data and "col" in data, f"Missing row or col for {item_name}"
        assert 1 <= data["row"] <= 7, f"Invalid row for {item_name}: {data['row']}"
        assert 1 <= data["col"] <= 3, f"Invalid col for {item_name}: {data['col']}"


def test_western_merchant_fixed_slot_matching():
    coords = yaml.safe_load(Path("config/ui_coordinates.yaml").read_text(encoding="utf-8"))
    items = coords["shop"]["western_merchant_items"]

    def find_slot(requested: str):
        for item_key, item_coords in items.items():
            is_m, sc = BotController._is_matching_shop_item(item_key, requested)
            if is_m or sc >= 0.80:
                return item_key, (item_coords["x"], item_coords["y"])
        return None, None

    # 1. Kim Cang Nguyên Thạch -> Row 7 Col 2
    matched_k, pos_k = find_slot("Kim cang nguyên thạch")
    assert matched_k == "Kim Cang Nguyên Thạch"
    assert pos_k == (375, 705)

    # 2. Linh Dược Tây Vực -> Row 1 Col 2
    matched_l, pos_l = find_slot("Linh dược Tây Vực")
    assert matched_l == "Linh Dược Tây Vực"
    assert pos_l == (375, 195)

    # 3. Tinh Kim Tây Vực -> Row 3 Col 1
    matched_t, pos_t = find_slot("Tinh Kim Tây Vực")
    assert matched_t == "Tinh Kim Tây Vực"
    assert pos_t == (285, 365)

    # 4. Hổ Phách Nguyên Thạch -> Row 6 Col 2
    matched_h, pos_h = find_slot("Hổ Phách Nguyên Thạch")
    assert matched_h == "Hổ Phách Nguyên Thạch"
    assert pos_h == (375, 620)

    # 5. Tuyết Tinh Nguyên Thạch -> Row 5 Col 2
    matched_tt, pos_tt = find_slot("Tuyết Tinh Nguyên Thạch")
    assert matched_tt == "Tuyết Tinh Nguyên Thạch"
    assert pos_tt == (375, 535)
