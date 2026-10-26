from pathlib import Path
import yaml
from src.text_recognition import is_same_city

def prioritize_spawns_for_city(spawns: list[dict], cur_city: str | None) -> list[dict]:
    if not cur_city:
        return spawns
    curr_spawns = []
    other_spawns = []
    for sp in spawns:
        if is_same_city(cur_city, sp.get("city", "")):
            curr_spawns.append(sp)
        else:
            other_spawns.append(sp)
    if curr_spawns:
        return curr_spawns + other_spawns
    return spawns


def test_merchant_prioritizes_current_city():
    spawns = [
        {"city": "Tuyền Châu", "x": 187, "y": 191},
        {"city": "Biện Kinh", "x": 180, "y": 185},
        {"city": "Thành Đô", "x": 225, "y": 225},
        {"city": "Tương Dương", "x": 187, "y": 180},
        {"city": "Dương Châu", "x": 206, "y": 194},
        {"city": "Đại Lý", "x": 178, "y": 187},
    ]

    # Case 1: Đang ở Biện Kinh
    res_bk = prioritize_spawns_for_city(spawns, "Biện Kinh")
    assert res_bk[0]["city"] == "Biện Kinh"
    assert len(res_bk) == len(spawns)

    # Case 2: Đang ở Thành Đô
    res_td = prioritize_spawns_for_city(spawns, "Thành Đô")
    assert res_td[0]["city"] == "Thành Đô"
    assert len(res_td) == len(spawns)

    # Case 3: Đang ở Tuyền Châu
    res_tc = prioritize_spawns_for_city(spawns, "Tuyền Châu")
    assert res_tc[0]["city"] == "Tuyền Châu"
    assert len(res_tc) == len(spawns)

    # Case 4: Đang ở Dương Châu (chữ chuẩn và OCR biến thể) -> Phải ưu tiên Dương Châu, KHÔNG nhầm sang Tương Dương
    res_dc = prioritize_spawns_for_city(spawns, "Dương Châu")
    assert res_dc[0]["city"] == "Dương Châu"

    res_dc_ocr = prioritize_spawns_for_city(spawns, "Bac mon Dudng Chau")
    assert res_dc_ocr[0]["city"] == "Dương Châu"

    # Case 5: Đang ở Tương Dương
    res_td2 = prioritize_spawns_for_city(spawns, "Tương Dương")
    assert res_td2[0]["city"] == "Tương Dương"

    # Case 6: Chuỗi mơ hồ "Duong" (không rõ Dương Châu hay Tương Dương) -> Không được đoán bừa match nhầm
    res_ambiguous = prioritize_spawns_for_city(spawns, "Duong")
    assert res_ambiguous[0]["city"] == "Tuyền Châu"  # Giữ nguyên thứ tự mặc định

    # Case 7: Không nhận diện được thành phố (None)
    res_none = prioritize_spawns_for_city(spawns, None)
    assert res_none[0]["city"] == "Tuyền Châu"  # giữ nguyên thứ tự ban đầu

    # Case 8: Thành phố không có trong danh sách spawn (ví dụ: Lâm An)
    res_other = prioritize_spawns_for_city(spawns, "Lâm An")
    assert res_other[0]["city"] == "Tuyền Châu"

    # Case 9: Đang ở map ngoại thành có hướng (Nam Thành Đô) -> Tuyệt đối KHÔNG nhầm là Thành Đô!
    res_nam_td = prioritize_spawns_for_city(spawns, "Nam Thành Đô")
    assert res_nam_td[0]["city"] == "Tuyền Châu"  # Giữ nguyên thứ tự mặc định, không ưu tiên Thành Đô

    assert is_same_city("Nam Thành Đô", "Thành Đô") is False
    assert is_same_city("Thành Đô", "Nam Thành Đô") is False
    assert is_same_city("Bắc Tuyền Châu", "Tuyền Châu") is False
    assert is_same_city("Đông Biện Kinh", "Biện Kinh") is False
    assert is_same_city("Thành Đô", "Thành Đô phủ") is True
    # Đại Lý and Đại Lý phủ refer to the same city (Đại Lý on world map/search, Đại Lý phủ on HUD)
    assert is_same_city("Đại Lý", "Đại Lý phủ") is True
    assert is_same_city("Đại Lý phủ", "Đại Lý") is True
    assert is_same_city("Đại Lý", "Đại Lý") is True


def test_click_map_search_result_dai_ly_strict():
    """Searching for 'Đại Lý' must only match exact 'Đại Lý', rejecting 'Đại Lý phủ' and 'Thành Đại Lý'."""
    from unittest.mock import MagicMock
    import numpy as np
    from src.bot_controller import BotController
    from src.text_recognition import TextBox

    bot = BotController.__new__(BotController)
    loader = MagicMock()
    loader.load_ui_coords.return_value = {
        "navigation": {
            "map_list_region": {"x1": 75, "y1": 250, "x2": 465, "y2": 865}
        }
    }
    bot._loader = loader
    bot._input = MagicMock()
    bot._screen = MagicMock()
    bot._text = MagicMock()

    dummy_frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._screen.capture.return_value = dummy_frame

    # 1. Boxes with only "Đại Lý phủ" -> Must return False
    bot._text.read.return_value = [TextBox("Đại Lý phủ", 0.95, 200, 300)]
    assert bot._click_map_search_result("Đại Lý") is False
    bot._input.tap.assert_not_called()

    # 2. Boxes with "Thành Đại Lý" -> Must return False
    bot._text.read.return_value = [TextBox("Thành Đại Lý", 0.95, 200, 300)]
    assert bot._click_map_search_result("Đại Lý") is False
    bot._input.tap.assert_not_called()

    # 3. Boxes with exact "Đại Lý" -> Must tap and return True
    bot._text.read.return_value = [
        TextBox("Đại Lý phủ", 0.95, 200, 300),
        TextBox("Đại Lý", 0.99, 200, 350),
    ]
    assert bot._click_map_search_result("Đại Lý") is True
    bot._input.tap.assert_called_once_with(200, 350)

