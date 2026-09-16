from src.text_recognition import (
    TextBox, GameTextRecognizer, fuzzy_score, normalize_game_text,
    is_explicit_quest_exhausted,
    is_western_merchant_name,
    is_safe_map_result,
)


def test_normalize_vietnamese():
    assert normalize_game_text("Chủ Thương Hội!") == "chu thuong hoi"


def test_fuzzy_typo_and_missing_accents():
    assert fuzzy_score("Chu Thuong Hoi", "Chủ thương hội") == 1.0
    assert fuzzy_score("Thuong nhan phuong tay", "Thương nhân phương Tây") > 0.9
    assert fuzzy_score(
        "Nhiem vu yeu cau tim Thuong nhan phuong tay de mua vat pham",
        "thương nhân phương tây",
    ) > 0.65


def test_best_dialog_option_ignores_other_buttons():
    boxes = [
        TextBox("Xem tinh hinh nhiem vu", .9, 800, 580),
        TextBox("Vi trieu dinh cong hien suc luc", .9, 400, 580),
    ]
    match = GameTextRecognizer.best_match(
        boxes, ["Vì triều đình cống hiến sức lực"], minimum=.5
    )
    assert match is not None
    assert match[1].center_x == 400


def test_daily_limit_information_is_not_completion():
    assert not is_explicit_quest_exhausted("Mỗi ngày có thể nhận tối đa 20 lần nhiệm vụ")


def test_explicit_daily_completion_is_detected():
    assert is_explicit_quest_exhausted("Hôm nay đã nhận tối đa 20 lần")


def test_western_merchant_requires_identity_tokens():
    assert is_western_merchant_name("Thương nhân Tây Vực")
    assert is_western_merchant_name("Thuong nhan Tay Ho")
    assert not is_western_merchant_name("Long Tuyen Thon")


def test_parent_city_does_not_match_directional_submap():
    assert is_safe_map_result("Thành Đô phủ", "Thành Đô")
    assert not is_safe_map_result("Nam Thành Đô", "Thành Đô")
