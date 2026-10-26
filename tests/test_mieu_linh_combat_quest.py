from src.quest_classifier import parse_yellow_entities, extract_combat_target

def test_mieu_linh_tuan_trai_combat_quest():
    tokens = ["Tan Thu", "Mieu Linh", "Tuan trai", "10 ten"]
    entities = parse_yellow_entities(tokens)
    assert entities["map_name"] == "Miêu Lĩnh"
    assert entities["monster"] == "Tuần trại"
    assert entities["count"] == 10

    text = "Tan Thu Auto ng hoi:Mieu Linh co bon Tuan trai vo co giet hai nhieu nguoi, hay di diet 10 ten . dua den noi lam nhiem vy khong?"
    c_map, c_monster, c_count = extract_combat_target(text, tokens)
    assert c_map == "Miêu Lĩnh"
    assert c_monster == "Tuần trại"
    assert c_count == 10
