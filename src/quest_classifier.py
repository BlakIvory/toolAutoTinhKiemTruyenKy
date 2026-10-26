"""
AutoQuestTool - Quest Classifier
Phân loại nhiệm vụ theo thứ tự:
1. Detect icon/template
2. Detect quest state
3. OCR (Phase 2)
4. Keyword/rule based
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path
import re
import numpy as np
from loguru import logger

from .state_machine import QuestType
from .vision_engine import VisionEngine
from .config_loader import AppConfig
from .text_recognition import fuzzy_score, normalize_game_text


KNOWN_DIALOGUE_CITIES = (
    "Tuyền Châu", "Biện Kinh", "Thành Đô", "Tương Dương",
    "Dương Châu", "Đại Lý", "Phượng Tường",
)

SPECIAL_CITY_ALIASES = {
    "thanh do phu": "Thành Đô",
    "phu thanh do": "Thành Đô",
    "thanh do": "Thành Đô",
    "thanh d8 phi": "Thành Đô",
    "thanh do phi": "Thành Đô",
    "thanh d8 phu": "Thành Đô",
    "thanh d0 phu": "Thành Đô",
    "thanh d8": "Thành Đô",
    "thanh d0": "Thành Đô",
    "thanh d0 phi": "Thành Đô",
    "bien kinh phu": "Biện Kinh",
    "phu bien kinh": "Biện Kinh",
    "bien kinh": "Biện Kinh",
    "tuyen chau phu": "Tuyền Châu",
    "phu tuyen chau": "Tuyền Châu",
    "tuyen chau": "Tuyền Châu",
    "tuong duong phu": "Tương Dương",
    "phu tuong duong": "Tương Dương",
    "tuong duong": "Tương Dương",
    "duong chau phu": "Dương Châu",
    "phu duong chau": "Dương Châu",
    "duong chau": "Dương Châu",
    "dai ly phu": "Đại Lý",
    "phu dai ly": "Đại Lý",
    "dai ly": "Đại Lý",
    "phuong tuong phu": "Phượng Tường",
    "phu phuong tuong": "Phượng Tường",
    "phuong tuong": "Phượng Tường",
}

KNOWN_SECTS = {
    "cai bang": "Cái Bang",
    "cai bang phai": "Cái Bang",
    "nga my": "Nga My",
    "nga mi": "Nga My",
    "nga my phai": "Nga My",
    "vo dang": "Võ Đang",
    "vo dang phai": "Võ Đang",
    "thieu lam": "Thiếu Lâm",
    "thieu lam phai": "Thiếu Lâm",
    "duong mon": "Đường Môn",
    "duong mon phai": "Đường Môn",
    "thien vuong": "Thiên Vương",
    "thien vuong bang": "Thiên Vương",
    "con lon": "Côn Lôn",
    "con lon phai": "Côn Lôn",
    "thuy yen": "Thúy Yên",
    "thuy yen mon": "Thúy Yên",
    "thien nhan": "Thiên Nhẫn",
    "thien nhan giao": "Thiên Nhẫn",
    "ngu doc": "Ngũ Độc",
    "ngu doc giao": "Ngũ Độc",
    "ngi doc": "Ngũ Độc",
    "ngi doc giao": "Ngũ Độc",
    "ngui dic": "Ngũ Độc",
    "ngui dic giao": "Ngũ Độc",
}

KNOWN_COMBAT_MAPS = {
    "tuong van dong": ("Cái Bang", "Tường Vân động"),
    "tuong van": ("Cái Bang", "Tường Vân động"),
    "bang tam dong": ("Nga My", "Băng Tâm động"),
    "bang tam": ("Nga My", "Băng Tâm động"),
    "duoc vuong dong": ("Ngũ Độc", "Dược Vương động"),
    "duoc vuong": ("Ngũ Độc", "Dược Vương động"),
    "thien vien dong": ("Thiếu Lâm", "Thiền Viên động"),
    "huyen vu dong": ("Võ Đang", "Huyền Vũ động"),
    "tang 2 duong trung dong": ("Hạnh Hoa thôn", "Tầng 2 Dương Trung động"),
    "tang 1 duong trung dong": ("Hạnh Hoa thôn", "Tầng 1 Dương Trung động"),
    "duong trung dong": ("Hạnh Hoa thôn", "Dương Trung động"),
    "duong trung": ("Hạnh Hoa thôn", "Dương Trung động"),
    "hanh hoa thon": ("Hạnh Hoa thôn", "Hạnh Hoa thôn"),
    "van mong trach": ("Ba Lăng huyện", "Vân Mộng trạch"),
    "van mong": ("Ba Lăng huyện", "Vân Mộng trạch"),
    "luong son bac": ("Lương Sơn Bạc", "Lương Sơn Bạc"),
    "luong son": ("Lương Sơn Bạc", "Lương Sơn Bạc"),
    "phuc nguu son": ("Phục Ngưu Sơn", "Phục Ngưu Sơn"),
    "phuc nguu": ("Phục Ngưu Sơn", "Phục Ngưu Sơn"),
    "thanh thanh son": ("Thanh Thành Sơn", "Thanh Thành Sơn"),
    "diem thuong son": ("Điểm Thương Sơn", "Điểm Thương Sơn"),
    "khoa lang dong": ("Khoa Lang động", "Khoa Lang động"),
    "mac cao quat": ("Mạc Cao Quật", "Mạc Cao Quật"),
    "mieu linh": ("Đại Lý", "Miêu Lĩnh"),
}

# Các thành lớn và từ ngữ đô thị/bối cảnh cần tách riêng, loại trừ khỏi map quái
MAJOR_CITIES_AND_URBAN = (
    "bien kinh", "thanh do", "tuong duong", "duong chau", "tuyen chau",
    "dai ly", "phuong tuong", "lam an", "trong thanh", "thanh thi",
    "phu thanh do", "phu bien kinh", "phu tuyen chau", "phu tuong duong",
    "phu duong chau", "phu dai ly", "phu phuong tuong", "chu thuong hoi"
)


def parse_sect_and_combat_map(raw_text: str, with_accents: bool = True) -> tuple[str, str]:
    """
    Phân tích tách biệt:
    - Địa bàn / Môn phái (ví dụ: 'Cái Bang', 'Nga My', 'Ngũ Độc')
    - Tên bản đồ săn quái (ví dụ: 'Tường Vân động', 'Ngũ Độc Mê Cung tầng 2')
    Đồng thời loại trừ hoàn toàn các thành lớn và từ ngữ đô thị để tránh nhầm lẫn.
    """
    norm = normalize_game_text(raw_text)
    if not norm:
        return "", ""

    # 1. Nếu là thành lớn hoặc từ ngữ đô thị -> Không phải map quái
    if any(norm == urban or norm.startswith(urban + " ") or norm.endswith(" " + urban) for urban in MAJOR_CITIES_AND_URBAN):
        return "", ""

    # 2. Khớp trực tiếp với bảng map quái môn phái đã biết
    for k_map, (sect, map_name) in KNOWN_COMBAT_MAPS.items():
        if k_map in norm:
            res_sect = sect if with_accents else normalize_game_text(sect)
            res_map = map_name if with_accents else normalize_game_text(map_name)
            return res_sect, res_map

    # 3. Khớp tiền tố môn phái (ví dụ: 'Cái Bang ...', 'Ngi Doc giao me cung tang 2')
    for s_norm, s_name in KNOWN_SECTS.items():
        if norm.startswith(s_norm):
            sub = norm[len(s_norm):].strip()
            sub = re.sub(r"^(?:o|tai|sau\s+nui\s+o|sau\s+nui|giao)\s+", "", sub).strip()
            if sub:
                if "me cung" in sub:
                    floor_m = re.search(r"(?:tang\s*(\d+)|me\s*cung\s*(\d+)|\b(\d+)\b)", sub)
                    floor_val = (floor_m.group(1) or floor_m.group(2) or floor_m.group(3)) if floor_m else ""
                    floor = f" tầng {floor_val}" if floor_val else ""
                    clean_map = f"{s_name} Mê Cung{floor}" if with_accents else f"{normalize_game_text(s_name)} me cung{floor}"
                else:
                    clean_map = " ".join(w.capitalize() if w != "dong" else "động" for w in sub.split()) if with_accents else sub
                res_sect = s_name if with_accents else normalize_game_text(s_name)
                return res_sect, clean_map

    # 4. Kiểm tra map mê cung (kể cả khi tên phái nằm giữa câu hoặc không nằm ở đầu)
    if "me cung" in norm:
        floor_m = re.search(r"(?:tang\s*(\d+)|me\s*cung\s*(\d+)|\b(\d+)\b)", norm)
        floor_val = (floor_m.group(1) or floor_m.group(2) or floor_m.group(3)) if floor_m else ""
        floor = f" tầng {floor_val}" if floor_val else ""
        found_sect = ""
        found_sect_norm = ""
        for s_norm, s_name in KNOWN_SECTS.items():
            if s_norm in norm:
                found_sect = s_name
                found_sect_norm = normalize_game_text(s_name)
                break
        if found_sect:
            clean_map = f"{found_sect} Mê Cung{floor}" if with_accents else f"{found_sect_norm} me cung{floor}"
            return found_sect if with_accents else found_sect_norm, clean_map
        clean_map = f"Mê Cung{floor}" if with_accents else f"me cung{floor}"
        return "", clean_map

    # 5. Kiểm tra hậu tố map quái (động, sơn, cốc, đảo, trạch, lĩnh, v.v.)
    # Trừ các tên quái kết thúc bằng trại/quân/tháp (ví dụ 'Tuần trại', 'Liêu quân')
    if norm in ("tuan trai", "tuan tra", "bon tuan trai", "lieu quan", "lieu binh", "quan lieu"):
        return "", ""
    if "KNOWN_MONSTER_ALIASES" in globals() and norm in globals()["KNOWN_MONSTER_ALIASES"]:
        return "", ""

    map_suffixes = ("dong", "son", "quang", "dao", "coc", "thon", "mo", "rung", "ho", "trai", "dien", "trang", "trach", "quan", "do", "dinh", "am", "thap", "linh")
    if any(norm.endswith(" " + kw) or norm == kw for kw in map_suffixes):
        return "", raw_text.strip() if with_accents else norm

    return "", ""


KNOWN_DIALOGUE_NPCS = (
    "Bách Hiểu Sinh",
    "Vệ binh thành môn",
    "Vệ Binh Thành Môn",
    "Vân Du đạo nhân",
    "Nạn dân",
    "Nạn dân Giáp",
    "Nạn dân Ất",
    "Nạn dân Bính",
    "Nạn dân Đinh",
    "Khẩu Chuẩn",
    "Khấu Chuẩn",
    "Phạm Trọng Yêm",
    "Vương Nghiệp Vi",
    "Tào Đại Lực",
    "Tào Bân",
    "Hành Cước Thương Nhân",
    "Dã Tẩu",
    "La Sa",
    "Lâm Dược Sư",
    "Thái thú",
    "Bộ đầu",
    "Tiểu đồng",
    "Ấu đồng",
    "Ấu Đồng",
    "Lão đài",
    "Hạ Kiệt",
    "Uy Nhĩ Tốn",
    "Chu Bân Cử",
    "Tiêu sư",
    "Sứ giả môn phái",
    "Quản lý cửa hiệu",
    "Đại Chí Cường",
    "Dụ Chân Nhân",
    "Thủ kho Bắc",
    "Thủ kho Nam",
    "Thủ kho Đông",
    "Thủ kho Tây",
    "Xa phu",
    "Xa phu Bắc",
    "Xa phu Nam",
    "Chủ Tạp hóa",
    "Chủ Tiệm Nam phục",
    "Chủ Tiệm Nữ phục",
    "Chủ Tiệm Vũ khí",
    "Chủ tiệm vũ khí",
    "Tiệm Vũ khí",
    "Tiệm vũ khí",
    "Chủ Kim hoàn",
    "Chủ tiền trang",
    "Võ sư",
    "Hòa Phụng",
    "Chu sứ giả",
    "Di tộc Sở quốc",
    "Thuyền phu Cô Đảo",
    "Cố Nhân Tương Phùng",
    "Tiểu Bạch",
    "Mạc Quân đạo trưởng",
    "Vạn Sự Thông",
    "Chỉ huy sứ",
    "Quan thành sứ",
    "Quan viên thoái nhiệm",
    "Sứ giả Tàng Kiếm",
    "Chủ tửu lầu",
    "Sứ giả bang phái",
    "Lễ quan",
    "Lý Phong Thu",
    "Cẩm Nương",
    "Cam Nương",
    "Du Phương đại phu",
    "Dư Nương",
    "Vương Cẩm",
    "Mạc Nhĩ",
    "Chủ Dược điếm",
    "Thầy bói",
    "Thầy Bói",
    "Cấm quân",
    "Nha dịch",
    "Nha Dịch",
)


def _is_shop_type_incompatible(words1: set[str], words2: set[str]) -> bool:
    """Kiểm tra hai tập từ có thuộc các danh mục cửa hiệu xung đột/khác nhau không."""
    categories = (
        {"vu", "khi", "khl", "vk"},
        {"nu", "ni", "nt"},
        {"nam"},
        {"kim", "hoan"},
        {"duoc", "diem", "thuoc"},
        {"tap", "hoa"},
        {"tien", "trang"},
        {"tuu", "tiu", "lau"},
    )
    for cat in categories:
        has1 = bool(words1 & cat)
        has2 = bool(words2 & cat)
        if has1 != has2 and any(w in words1 or w in words2 for w in ("tiem", "chu")):
            return True
    return False


NPC_ALIASES = {
    "chu tiem vu khi": "Chủ Tiệm Vũ khí",
    "chu tiem vu khl": "Chủ Tiệm Vũ khí",
    "chu tiem vk": "Chủ Tiệm Vũ khí",
    "tiem vu khi": "Chủ Tiệm Vũ khí",
    "tiem vu khl": "Chủ Tiệm Vũ khí",
    "tiem vk": "Chủ Tiệm Vũ khí",
    "vu khi": "Chủ Tiệm Vũ khí",
    "vu khl": "Chủ Tiệm Vũ khí",
    "chủ tiệm vũ khí": "Chủ Tiệm Vũ khí",
    "tiệm vũ khí": "Chủ Tiệm Vũ khí",
    "thay boi": "Thầy bói",
    "thayboi": "Thầy bói",
    "thay": "Thầy bói",
    "thay bdi": "Thầy bói",
    "thay bai": "Thầy bói",
    "chu tuu lau": "Chủ tửu lầu",
    "chu tiu lau": "Chủ tửu lầu",
    "chi tiu lau": "Chủ tửu lầu",
    "cho ti lau": "Chủ tửu lầu",
    "cho tiu lau": "Chủ tửu lầu",
    "tuu lau": "Chủ tửu lầu",
    "xa phu": "Xa phu",
    "nan dan": "Nạn dân",
    "nạn dân": "Nạn dân",
    "nan dan giap": "Nạn dân Giáp",
    "nan dan at": "Nạn dân Ất",
    "nan dan binh": "Nạn dân Bính",
    "nan dan dinh": "Nạn dân Đinh",
    # Tên NPC trong game là "Tiêu sư" (người hộ tống), không phải
    # "Tiêu sứ". Hai từ đều bị normalize thành "tieu su", nên phải
    # ép về tên chuẩn trước khi gõ vào minimap.
    "tieu su": "Tiêu sư",
    "cam nuong": "Cẩm Nương",
    "can nuong": "Cẩm Nương",
    "cam nong": "Cẩm Nương",
    "cam nuog": "Cẩm Nương",
    "du phuong dai phu": "Du Phương đại phu",
    "du phuong": "Du Phương đại phu",
    "duong dai phu": "Du Phương đại phu",
    "duong nuong": "Dư Nương",
    "du nuong": "Dư Nương",
    "vuong cam": "Vương Cẩm",
    "mac nhi": "Mạc Nhĩ",
    "duoc diem": "Chủ Dược điếm",
    "chu duoc diem": "Chủ Dược điếm",
    "quan vien thoai nhiem": "Quan viên thoái nhiệm",
    "quan vien thoai niem": "Quan viên thoái nhiệm",
    "quan vien thoai nhlem": "Quan viên thoái nhiệm",
    "chu tiem trang phuc nu": "Chủ Tiệm Nữ phục",
    "chu tiem trang phuc nam": "Chủ Tiệm Nam phục",
    "chu tiem trang phyc ni": "Chủ Tiệm Nữ phục",
    "chu tiem trang phyc nt": "Chủ Tiệm Nữ phục",
    "chi tiem trang phyc ni": "Chủ Tiệm Nữ phục",
    "chi tiem trang phyc nt": "Chủ Tiệm Nữ phục",
    "chi tiem trang phuc nu": "Chủ Tiệm Nữ phục",
    "chi tiem trang phuc nam": "Chủ Tiệm Nam phục",
    "chi tiem nam phuc": "Chủ Tiệm Nam phục",
    "chi tiem nam phyc": "Chủ Tiệm Nam phục",
    "chi tiem nu phuc": "Chủ Tiệm Nữ phục",
    "chi tiem nu phyc": "Chủ Tiệm Nữ phục",
    "tiem nu phuc": "Chủ Tiệm Nữ phục",
    "tiem nam phuc": "Chủ Tiệm Nam phục",
    "chu tiem nu phuc": "Chủ Tiệm Nữ phục",
    "chu tiem nam phuc": "Chủ Tiệm Nam phục",
    "chu tiem nu phyc": "Chủ Tiệm Nữ phục",
    "chu tiem nam phyc": "Chủ Tiệm Nam phục",
    "tiem nu phyc": "Chủ Tiệm Nữ phục",
    "tiem nam phyc": "Chủ Tiệm Nam phục",
    "cam quan": "Cấm quân",
    "can quan": "Cấm quân",
    "chu ban cu": "Chu Bân Cử",
    "chu ban cy": "Chu Bân Cử",
    "chu ban cii": "Chu Bân Cử",
    "chu ban": "Chu Bân Cử",
    "ban cu": "Chu Bân Cử",
    "chu bàn cử": "Chu Bân Cử",
    "chủ bàn cử": "Chu Bân Cử",
    "chủ bân cử": "Chu Bân Cử",
    "chu bân cử": "Chu Bân Cử",
    "au dong": "Ấu đồng",
    "au dong.": "Ấu đồng",
    "au dpng": "Ấu đồng",
    "au dong,": "Ấu đồng",
    "ấu đồng": "Ấu đồng",
    "ấu đồng.": "Ấu đồng",
    "nha dich": "Nha dịch",
    "nhadich": "Nha dịch",
    "nha dlch": "Nha dịch",
    "nha djch": "Nha dịch",
    "nha dich.": "Nha dịch",
    "nha dịch": "Nha dịch",
    "tao dai luc": "Tào Đại Lực",
    "tao dai luic": "Tào Đại Lực",
    "tao dai lyc": "Tào Đại Lực",
    "taodai luic": "Tào Đại Lực",
    "taodai luc": "Tào Đại Lực",
    "taodai lyc": "Tào Đại Lực",
    "tào đại lực": "Tào Đại Lực",
    "tao ban": "Tào Bân",
    "tao ban.": "Tào Bân",
    "tao ban:": "Tào Bân",
    "tao bân": "Tào Bân",
    "tào bán": "Tào Bân",
    "tào bản": "Tào Bân",
    "tào bân": "Tào Bân",
}


def get_canonical_npc_name(npc: str, original_text: str = "") -> str:
    """Restore proper Vietnamese accents and capitalization for known NPC names."""
    if not npc:
        return ""
    norm_npc = normalize_game_text(npc)
    if norm_npc in NPC_ALIASES:
        return NPC_ALIASES[norm_npc]

    norm_words = set(norm_npc.split())
    has_nam = "nam" in norm_words
    has_nu = bool(norm_words & {"nu", "ni", "nt"})
    has_refugee_sub = bool(norm_words & {"giap", "at", "binh", "dinh"})

    scored_aliases = []
    for alias, canonical in NPC_ALIASES.items():
        a_words = set(alias.split())
        c_words = set(normalize_game_text(canonical).split())
        if _is_shop_type_incompatible(norm_words, a_words) or _is_shop_type_incompatible(norm_words, c_words):
            continue
        if has_nam and (c_words & {"nu", "ni", "nt"} or a_words & {"nu", "ni", "nt"}):
            continue
        if has_nu and ("nam" in c_words or "nam" in a_words):
            continue
        if alias in ("can quan", "cam quan") or canonical == "Cấm quân":
            if not (norm_words & {"cam", "can", "kam"} and (norm_words & {"quan", "qyan", "qun"} or "quan" in norm_npc)):
                continue
        # Tránh nhầm lẫn Nạn dân với Nạn dân Giáp/Ất/Bính/Đinh
        a_has_sub = bool(a_words & {"giap", "at", "binh", "dinh"})
        if has_refugee_sub != a_has_sub and ("nan dan" in alias or "nan dan" in norm_npc):
            continue
        sc = fuzzy_score(norm_npc, alias)
        if alias == norm_npc or (alias in norm_npc and len(alias) >= 8) or sc >= 0.75:
            scored_aliases.append((sc, canonical))
    if scored_aliases:
        return max(scored_aliases, key=lambda x: x[0])[1]

    for known in KNOWN_DIALOGUE_NPCS:
        if normalize_game_text(known) == norm_npc:
            return known

    # Fuzzy match with strict threshold and Nam vs Nữ guard
    scored = []
    for k in KNOWN_DIALOGUE_NPCS:
        norm_k = normalize_game_text(k)
        k_words = set(norm_k.split())
        if _is_shop_type_incompatible(norm_words, k_words):
            continue
        if has_nam and (k_words & {"nu", "ni", "nt"}):
            continue
        if has_nu and ("nam" in k_words):
            continue
        if k == "Cấm quân" or norm_k in ("cam quan", "can quan"):
            if not (norm_words & {"cam", "can", "kam"} and (norm_words & {"quan", "qyan", "qun"} or "quan" in norm_npc)):
                continue
        k_has_sub = bool(k_words & {"giap", "at", "binh", "dinh"})
        if has_refugee_sub != k_has_sub and ("nan dan" in norm_k or "nan dan" in norm_npc):
            continue
        sc = fuzzy_score(norm_npc, norm_k)
        if sc >= 0.75:
            scored.append((sc, k))
    if scored:
        return max(scored, key=lambda x: x[0])[1]

    if original_text:
        import re
        orig_match = re.search(
            r"(?:gặp|tìm|nói với|đối thoại với|thông báo cho|báo tin cho|báo cho)\s+([^:,.!?\n\r]+?)(?=\s+nói|\s+tại|\s+để|\s+rồi|\s+và|\s+báo|\s+hàng|:|,|\.|$)",
            original_text, re.IGNORECASE
        )
        if orig_match and orig_match.group(1).strip():
            cand = orig_match.group(1).strip()
            norm_cand = normalize_game_text(cand)
            if norm_cand in NPC_ALIASES:
                return NPC_ALIASES[norm_cand]
            for known in KNOWN_DIALOGUE_NPCS:
                if normalize_game_text(known) == norm_cand or fuzzy_score(norm_cand, normalize_game_text(known)) >= 0.75:
                    return known
            return cand

    return npc.title()


def extract_dialogue_target(text: str, with_accents: bool = False, yellow_tokens: Optional[list[str]] = None) -> tuple[str, str]:
    """
    Extract approximate city/NPC names from a commerce quest description.
    If with_accents=True, restores proper Vietnamese diacritics and capitalization for NPC.
    """
    import re

    PRONOUNS = {
        "ong ta", "ba ta", "han", "han ta", "nguoi do", "ho", "ai do",
        "ong ay", "ba ay", "nguoi", "ket thuc", "ket thuc doi thoai", "thuc doi thoai",
    }

    yellow_city = ""
    yellow_npc = ""
    if yellow_tokens:
        entities = parse_yellow_entities(yellow_tokens)
        yellow_city = entities.get("city") or entities.get("map_name") or ""
        yellow_npc = entities.get("npc") or ""
        # Chỉ trả ngay khi chữ vàng đã cung cấp đủ cả hai thực thể.
        # Trước đây, chỉ cần nhận ra thành phố là hàm trả về ngay,
        # làm mất tên NPC mới dù full text OCR vẫn đọc rõ "nói với ...".
        if yellow_city and yellow_npc:
            if with_accents:
                yellow_npc = get_canonical_npc_name(yellow_npc, original_text=text)
            return yellow_city, yellow_npc

    normalized = normalize_game_text(text)

    # Nếu văn bản chứa dấu hiệu của nhiệm vụ Mua hàng Tây Vực (chứa 'thương nhân tây vực', 'để tìm ông ta', hoặc liệt kê nhiều thành phố)
    # thì tuyệt đối KHÔNG trích xuất mục tiêu đối thoại NPC để tránh đi lạc
    city_count = sum(1 for c in ("bien kinh", "thanh do", "tuyen chau", "tuong duong", "duong chau", "dai ly") if c in normalized)
    is_western_merchant_text = (
        "thuong nhan tay vuc" in normalized
        or "thuong nhan phuong tay" in normalized
        or "de tim ong ta" in normalized
        or "linh duoc" in normalized
        or (("thuong nhan" in normalized or "tay vuc" in normalized) and any(w in normalized for w in ("dat", "mua", "lay", "de tim")))
        or city_count >= 4
    )
    if is_western_merchant_text:
        return "", ""

    city = ""
    # Mẫu tìm thành phố / bản đồ cần đến (loại trừ 'vô tội' bị nhận nhầm thành 'tới')
    city_match = re.search(
        r"(?<!\bvo\s)(?:\bhay den\b|\bden\b|\btoi\b)\s+([a-z0-9 ]+?)(?=\s+noi voi|\s+gap|\s+tim|\s+thong bao|\s+bao tin|\s+bao cho|\s+de|\s+hoi|,|\.|$)",
        normalized
    )
    if city_match:
        city = city_match.group(1).strip()

    if not city:
        for pattern in (
            r"(?<!\bvo\s)(?:\bden\b|\btoi\b)\s+([a-z0-9 ]+?)(?=\s+noi voi|\s+gap|\s+tim|\s+thong bao|\s+de|$)",
            r"tai\s+([a-z0-9 ]+?)(?=\s+noi voi|\s+gap|\s+tim|\s+thong bao|\s+co|$)",
        ):
            match = re.search(pattern, normalized)
            if match:
                city = match.group(1).strip()
                break

    # Canonicalize known cities / special place names (e.g. Thành Đô Phủ -> Thành Đô)
    norm_city = normalize_game_text(city)
    # Loại bỏ các từ hành động/chiến đấu/nhiệm vụ bị lọt vào chuỗi tên thành phố
    if any(bad in norm_city for bad in ("diet", "danh", "giet", "cung voi", "doi thoai", "hoan thanh", "nhiem vu", "vo toi")):
        city = ""
        norm_city = ""

    if norm_city in SPECIAL_CITY_ALIASES:
        city = SPECIAL_CITY_ALIASES[norm_city]
    else:
        matched_alias = None
        for alias in sorted(SPECIAL_CITY_ALIASES.keys(), key=len, reverse=True):
            if alias in normalized:
                city = SPECIAL_CITY_ALIASES[alias]
                matched_alias = True
                break
        if not matched_alias and normalized:
            scored = [(fuzzy_score(normalized, name), name) for name in KNOWN_DIALOGUE_CITIES]
            score, known_city = max(scored, key=lambda item: item[0])
            if score >= .48:
                city = known_city
            elif len(norm_city) > 20:
                city = ""

    npc = ""
    # Mẫu "cùng với [NPC] đối thoại / hoàn thành nhiệm vụ" (ưu tiên cao nhất vì là mục tiêu thực thi trực tiếp)
    cung_voi_idx = normalized.find("cung voi")
    if cung_voi_idx >= 0:
        after_cung_voi = normalized[cung_voi_idx + len("cung voi"):].strip()
        for alias, canon in sorted(NPC_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
            if alias in after_cung_voi:
                npc = canon if with_accents else alias
                break
        if not npc:
            for known_npc in KNOWN_DIALOGUE_NPCS:
                norm_k = normalize_game_text(known_npc)
                if norm_k in after_cung_voi:
                    npc = known_npc if with_accents else norm_k
                    break
        if not npc:
            cv_match = re.search(
                r"^([a-z0-9 ]+?)(?=\s+doi\s*thoai|\s+hoan\s*thanh|\s+de|\s+tai|,|\.|$)",
                after_cung_voi
            )
            if cv_match:
                npc = cv_match.group(1).strip()

    # Mẫu tìm tên NPC: gap [Tên NPC] noi ... hoặc thong bao cho [Tên NPC]
    # Nếu chưa tìm thấy NPC hoặc NPC đang bị gán là "xa phu" do dòng chỉ dẫn chuyển thành "cùng với Xa phu đối thoại"
    # thì ưu tiên tìm NPC đích thực sự được nhắc đến trong nội dung chính ("thông báo cho ấu đồng", "gặp ...", "nói với ...")
    main_search_text = normalized[:cung_voi_idx].strip() if cung_voi_idx >= 0 else normalized
    norm_npc = normalize_game_text(npc)
    if not npc or norm_npc == "xa phu":
        for search_src in (main_search_text, normalized):
            npc_match = re.search(
                r"(?:gap|tim|noi voi|doi thoai voi|thong bao cho|bao tin cho|bao cho|nhan tin cho|nhac nho|dua tin cho)\s+([a-z0-9 ]+?)(?=\s+cung\s*voi|\s+cung|\s+noi|\s+tai|\s+de|\s+roi|\s+va|\s+bao|\s+hang|:|,|\.|$)",
                search_src
            )
            if npc_match:
                cand = npc_match.group(1).strip()
                if cand and cand not in PRONOUNS:
                    for a_norm in sorted(NPC_ALIASES.keys(), key=len, reverse=True):
                        if a_norm in cand:
                            cand = a_norm
                            break
                    npc = cand
                    break
            else:
                for pattern in (
                    r"(?:noi|doi thoai) voi ([a-z0-9 ]+?)(?= cung | hang | de | roi | va | tai |:|$)",
                    r"(?:gap|tim|thong bao cho) ([a-z0-9 ]+?)(?= cung | tai | de | roi | va |:|$)",
                ):
                    match = re.search(pattern, search_src)
                    if match:
                        cand = match.group(1).strip()
                        if cand and cand not in PRONOUNS:
                            for a_norm in sorted(NPC_ALIASES.keys(), key=len, reverse=True):
                                if a_norm in cand:
                                    cand = a_norm
                                    break
                            npc = cand
                            break
            if npc and normalize_game_text(npc) != "xa phu":
                break

    # Lọc bỏ các đại từ xưng hô, nút giao diện không phải tên NPC
    norm_npc = normalize_game_text(npc)
    if norm_npc in PRONOUNS or any(norm_npc.startswith(p) for p in ("ong ta", "ba ta", "han ta", "ket thuc")):
        npc = ""

    # Chữ vàng là nguồn tin cậy hơn OCR toàn dòng, nhưng nếu nó
    # thiếu một trường thì giữ kết quả regex từ full text cho trường còn lại.
    city = yellow_city or city
    npc = yellow_npc or npc

    if with_accents and npc:
        npc = get_canonical_npc_name(npc, original_text=text)

    return city, npc


def is_auto_abandon_dialogue_quest(npc: str = "", text: str = "") -> bool:
    """
    Kiểm tra xem nhiệm vụ đối thoại có thuộc danh mục cần tự động hủy hay không
    (Ví dụ: NPC 'Ấu đồng' ở Tuyền Châu theo yêu cầu người dùng).
    """
    norm_n = normalize_game_text(npc)
    norm_t = normalize_game_text(text)
    if any(bad in norm_n for bad in ("au dong", "au dpng")):
        return True
    if any(k in norm_t for k in ("thong bao cho au dong", "gap au dong", "noi voi au dong", "cho au dong", "cho au dpng")):
        return True
    return False


def clean_shop_item_name(item: str) -> str:
    """Loại bỏ các hậu tố/tiền tố giao diện, tiêu đề nhiệm vụ dính vào tên món đồ."""
    if not item:
        return ""
    import re
    cleaned = item.strip()
    # Loại bỏ các đuôi tiêu đề nhiệm vụ như: 'y Thuong hoi', 'ý Thương hội', 'Thương hội', 'Tuần hoàn', '[Thương hội]'
    # LƯU Ý: Không cắt nếu là 'Chủ thương hội' (chủ thương hội là NPC, không phải đuôi giao diện)
    cleaned = re.sub(r"(?i)\s*(?<!ch[uủ]\s)(?:[yýv\|\-–]?\s*th[uư][oơ]ng\s*h[oộ]i|tu[aà]n\s*ho[aà]n|nhi[eệ]m\s*v[uụ]).*$", "", cleaned).strip()
    cleaned = re.sub(r"(?i)\s+[yýv]$", "", cleaned).strip()
    # Loại bỏ tiền tố số lượng/từ chỉ loại: '1 ', '1 viên ', 'một sợi ', 'món ', 'cái ', 'miếng '...
    cleaned = re.sub(r"^(?:[0-9]+\s+|m[oộ]t\s+|hai\s+|ba\s+)?(?:c[aá]i|vi[eê]n|b[iì]nh|cu[oô]n|t[aấ]m|m[oó]n|h[oộ]p|g[oó]i|chi[eế]c|s[oợ]i|b[oộ]|mi[eế]ng|kh[oố]i|c[aâ]n|l[aạ]ng)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^(?:[0-9]+\s+|m[oộ]t\s+|hai\s+|ba\s+|c[aá]i\s+|vi[eê]n\s+|b[iì]nh\s+|cu[oô]n\s+|t[aấ]m\s+|m[oó]n\s+|h[oộ]p\s+|g[oó]i\s+|chi[eế]c\s+|s[oợ]i\s+|b[oộ]\s+|mi[eế]ng\s+|kh[oố]i\s+|c[aâ]n\s+|l[aạ]ng\s+)", "", cleaned, flags=re.IGNORECASE).strip()
    # Loại bỏ 'ở chỗ thương nhân...', 'tại thương nhân...', 'ở thương nhân...' nếu còn sót lại ở cuối
    cleaned = re.sub(r"(?i)\s*(?:[0o]\s*ch[oõọ]|t[aạ]i|t[uừ]|[0o])\s+th[uư][oơ]ng\s*nh[aâ]n.*$", "", cleaned).strip()
    return cleaned.strip()


# Whitelist matching the actual Western Merchant grid in ui_coordinates.yaml.
# A quest is a shop quest only when its requested item resolves to this list.
WESTERN_MERCHANT_ITEMS = (
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
    "Ám Phách Nguyên Thạch",
    "Dạ Quang Bôi",
    "Tuyết Tinh Nguyên Thạch",
    "Thanh Kim Nguyên Thạch",
    "Hổ Phách Nguyên Thạch",
    "Lông Lạc Đà",
    "Kim Cang Nguyên Thạch",
)


def match_western_merchant_item(raw_text: str) -> str:
    """Return a canonical merchant item only for a confident whitelist match."""
    if not raw_text:
        return ""
    from .text_recognition import normalize_game_text, fuzzy_score

    raw_norm = normalize_game_text(raw_text).strip()
    if not raw_norm:
        return ""

    # 0. Nếu raw_text chứa trực tiếp tên món đồ trong danh mục Thương nhân Tây Vực
    has_uyen_raw = "ho uyen" in raw_norm or "hj uyen" in raw_norm or "uyen" in raw_norm.split()
    sorted_catalog = sorted(WESTERN_MERCHANT_ITEMS, key=lambda x: len(normalize_game_text(x)), reverse=True)
    for item in sorted_catalog:
        item_norm = normalize_game_text(item)
        if item_norm in ("long lac da", "thit lac da") and (has_uyen_raw or "bao tay" in raw_norm):
            continue
        if item_norm == "ho uyen long lac da" and not has_uyen_raw:
            continue
        if item_norm in raw_norm:
            return item

    norm = normalize_game_text(clean_shop_item_name(raw_text)).strip()
    if not norm:
        return ""

    # 1. So khớp chính xác tuyệt đối với danh mục trước
    for item in WESTERN_MERCHANT_ITEMS:
        if norm == normalize_game_text(item):
            return item

    aliases = {
        "thit lac da": "Thịt Lạc Đà",
        "mieng thit lac da": "Thịt Lạc Đà",
        "long lac da": "Lông Lạc Đà",
        "soi long lac da": "Lông Lạc Đà",
        "mot soi long lac da": "Lông Lạc Đà",
        "am yen nguyen thach": "Ám Yên Nguyên Thạch",
        "am phach nguyen thach": "Ám Phách Nguyên Thạch",
        "am pach nguyen thach": "Ám Phách Nguyên Thạch",
        "ho uyen long lac da": "Hộ Uyển Lông Lạc Đà",
        "ho uyen lac da": "Hộ Uyển Lông Lạc Đà",
        "hj uyen long lac da": "Hộ Uyển Lông Lạc Đà",
        "hj uyen lac da": "Hộ Uyển Lông Lạc Đà",
        "ho uyen": "Hộ Uyển Lông Lạc Đà",
        "hj uyen": "Hộ Uyển Lông Lạc Đà",
        "ho uyen long": "Hộ Uyển Lông Lạc Đà",
        "hj uyen long": "Hộ Uyển Lông Lạc Đà",
        "da nhung thi thao": "Hộ Uyển Lông Lạc Đà",
        "da nhung tha thao": "Hộ Uyển Lông Lạc Đà",
        "da nhung thao": "Hộ Uyển Lông Lạc Đà",
        "da quang boi": "Dạ Quang Bôi",
    }
    if norm in aliases:
        return aliases[norm]
    for alias_key, canonical in sorted(aliases.items(), key=lambda x: len(x[0]), reverse=True):
        if alias_key in norm:
            return canonical

    norm_words = norm.split()
    norm_word_set = set(norm_words)

    # Kiểm tra biến thể OCR đặc biệt của Hộ Uyển Lông Lạc Đà
    has_uyen = "ho uyen" in norm or "hj uyen" in norm or "uyen" in norm_word_set
    has_camel = "lac da" in norm or "long lac da" in norm
    if has_uyen and has_camel:
        return "Hộ Uyển Lông Lạc Đà"

    best_name = ""
    best_score = 0.0

    # Ưu tiên duyệt các vật phẩm có tên dài/cụ thể trước (vd: Hộ Uyển Lông Lạc Đà trước Lông Lạc Đà)
    sorted_catalog = sorted(WESTERN_MERCHANT_ITEMS, key=lambda x: len(normalize_game_text(x)), reverse=True)

    for item in sorted_catalog:
        item_norm = normalize_game_text(item)
        # Nếu item ngắn (như "Lông Lạc Đà") nhưng input có chứa "uyển", không được nhận nhầm
        if item_norm in ("long lac da", "thit lac da") and (has_uyen or "bao tay" in norm):
            continue
        # Ngược lại: nếu item là "Hộ Uyển Lông Lạc Đà" nhưng input KHÔNG CÓ "uyển", tuyệt đối không nhận nhầm khi chỉ có "lông lạc đà"
        if item_norm == "ho uyen long lac da" and not has_uyen:
            continue

        # Full quest text may contain the canonical item as a phrase.
        if item_norm in norm or (len(norm) >= 6 and norm in item_norm):
            return item
        # OCR đôi khi đảo thứ tự cụm danh từ: "Nguyên Thạch Hổ Phách"
        # thay vì "Hổ Phách Nguyên Thạch". Yêu cầu đủ toàn bộ từ trong tên
        # whitelist để không nhầm giữa các loại nguyên thạch.
        item_words = item_norm.split()
        if (
            len(item_words) >= 3
            and len(norm_words) <= len(item_words) + 8
            and set(item_words).issubset(norm_word_set)
        ):
            return item
        score = fuzzy_score(norm, item_norm)
        if score > best_score:
            best_name, best_score = item, score
    return best_name if best_score >= 0.78 else ""


def extract_yellow_text_boxes(frame: np.ndarray, region: Optional[tuple[int, int, int, int]] = None) -> list:
    """
    Trích xuất toàn bộ các cụm text màu VÀNG từ ảnh (cả hộp thoại lẫn bảng nhiệm vụ).
    Màu vàng trong không gian HSV: Hue 18..35, Saturation 90..255, Value 140..255.
    """
    if frame is None:
        return []
    import cv2
    from .text_recognition import GameTextRecognizer, TextBox

    x1 = y1 = 0
    roi = frame
    if region:
        rx1, ry1, rx2, ry2 = region
        h, w = frame.shape[:2]
        # Match GameTextRecognizer: UI regions are defined on a 1600x900 frame.
        if rx2 > w or ry2 > h or (w >= 1500 and h >= 800):
            x1 = max(0, min(w, int(rx1 * w / 1600)))
            y1 = max(0, min(h, int(ry1 * h / 900)))
            x2 = max(x1, min(w, int(rx2 * w / 1600)))
            y2 = max(y1, min(h, int(ry2 * h / 900)))
        else:
            x1 = max(0, min(w, rx1))
            y1 = max(0, min(h, ry1))
            x2 = max(x1, min(w, rx2))
            y2 = max(y1, min(h, ry2))
        roi = frame[y1:y2, x1:x2]

    if roi.size == 0 or len(roi.shape) != 3:
        return []

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, (18, 90, 140), (35, 255, 255))
    if np.sum(yellow_mask > 0) < 20:
        return []

    mask_3ch = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)
    # Upscale 2x để OCR nhận dạng tốt các ký tự/chữ nhỏ màu vàng
    mask_2x = cv2.resize(mask_3ch, (mask_3ch.shape[1] * 2, mask_3ch.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    ocr = GameTextRecognizer()
    boxes = ocr.read(mask_2x)

    adjusted = [
        TextBox(b.text.strip(), b.confidence, int(b.center_x / 2) + x1, int(b.center_y / 2) + y1)
        for b in boxes
        if b.text.strip()
    ]
    # Sắp xếp theo thứ tự đọc: từ trên xuống dưới, từ trái sang phải
    adjusted.sort(key=lambda b: (b.center_y // 20, b.center_x))
    return adjusted


KNOWN_MONSTER_ALIASES = {
    "ca sau": "Cá sấu",
    "ca sauu": "Cá sấu",
    "nu kiem khach": "Nữ Kiếm khách",
    "nt kiem khach": "Nữ Kiếm khách",
    "ni kiem khach": "Nữ Kiếm khách",
    "n kiem khach": "Nữ Kiếm khách",
    "coc doc": "Cóc độc",
    "goc dpc": "Cóc độc",
    "coc dpc": "Cóc độc",
    "goc doc": "Cóc độc",
    "coc dop": "Cóc độc",
    "goc dop": "Cóc độc",
    "bat phu ha dong su": "Bát phụ Hà Đông Sư",
    "bat phy ha dor": "Bát phụ Hà Đông Sư",
    "bat phu ha dor": "Bát phụ Hà Đông Sư",
    "bat phy ha dong su": "Bát phụ Hà Đông Sư",
    "bat phy ha": "Bát phụ Hà Đông Sư",
    "ha dong su": "Bát phụ Hà Đông Sư",
    "ha dor": "Bát phụ Hà Đông Sư",
    "bat phu": "Bát phụ Hà Đông Sư",
    "bat phy": "Bát phụ Hà Đông Sư",
    "ban nuong": "Bán nương",
    "barnh nuong": "Bán nương",
    "nguoi lun": "Người lùn",
    "nguoilun": "Người lùn",
    "tuyet quai": "Tuyết quái",
    "nhen doc": "Nhện độc",
    "thuc nhan thao": "Thực nhân thảo",
    "doc nhan thao": "Độc nhân thảo",
    "soi hoang": "Sói hoang",
    "soi dien": "Sói điên",
    "hac ho": "Hắc hổ",
    "bach ho": "Bạch hổ",
    "lieu quan": "Liêu quân",
    "lieu binh": "Liêu binh",
    "quan lieu": "Quân Liêu",
    "phan tac": "Phản tặc",
    "tho phi": "Thổ phỉ",
    "son tac": "Sơn tặc",
    "dao tac": "Đạo tặc",
    "hai tac": "Hải tặc",
    "phi tac": "Phi tặc",
    "gian te": "Gián tế",
    "xi hoa hac giao do": "Xí Hỏa Hắc giáo đồ",
    "xi hoa bach giao do": "Xí Hỏa Hắc giáo đồ",
    "xi hoa giao do": "Xí Hỏa Hắc giáo đồ",
    "hac giao do": "Xí Hỏa Hắc giáo đồ",
    "bach giao do": "Xí Hỏa Hắc giáo đồ",
    "hi hoa hac giao do": "Xí Hỏa Hắc giáo đồ",
    "hi hoa bach giao do": "Xí Hỏa Hắc giáo đồ",
    "giao do": "Xí Hỏa Hắc giáo đồ",
    "tuan trai": "Tuần trại",
    "tuan tra": "Tuần trại",
    "bon tuan trai": "Tuần trại",
}

CUSTOM_ITEM_ALIASES: dict[str, str] = {}
CUSTOM_MAP_ALIASES: dict[str, str] = {}


def load_custom_dictionary(dict_file: Optional[Path | str] = None) -> int:
    """
    Nạp danh từ định nghĩa từ custom_dictionary.yaml (NPC, quái vật, bản đồ, vật phẩm)
    vào các bảng aliases và whitelist để tránh OCR/nhận diện sai.
    """
    global KNOWN_DIALOGUE_NPCS
    if dict_file is None:
        dict_file = Path(__file__).resolve().parent.parent / "config" / "custom_dictionary.yaml"
    else:
        dict_file = Path(dict_file)

    if not dict_file.exists():
        return 0

    try:
        import yaml
        with open(dict_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        entities = data.get("custom_entities", [])
        known_npcs_list = list(KNOWN_DIALOGUE_NPCS)
        known_npcs_set = set(known_npcs_list)

        count = 0
        for item in entities:
            if not isinstance(item, dict):
                continue
            cat = str(item.get("category", "")).strip().lower()
            canonical = str(item.get("canonical_name", "")).strip()
            aliases_raw = item.get("aliases", "")
            if not canonical:
                continue

            if isinstance(aliases_raw, list):
                aliases = [str(a).strip() for a in aliases_raw if str(a).strip()]
            else:
                aliases = [a.strip() for a in str(aliases_raw).split(",") if a.strip()]

            all_forms = set(aliases) | {canonical}

            if cat in ("npc",):
                if canonical not in known_npcs_set:
                    known_npcs_list.append(canonical)
                    known_npcs_set.add(canonical)
                for form in all_forms:
                    norm = normalize_game_text(form)
                    if norm:
                        NPC_ALIASES[norm] = canonical
                count += 1

            elif cat in ("quái vật", "quai vat", "monster"):
                for form in all_forms:
                    norm = normalize_game_text(form)
                    if norm:
                        KNOWN_MONSTER_ALIASES[norm] = canonical
                count += 1

            elif cat in ("bản đồ", "ban do", "map"):
                for form in all_forms:
                    norm = normalize_game_text(form)
                    if norm:
                        CUSTOM_MAP_ALIASES[norm] = canonical
                        if norm not in KNOWN_COMBAT_MAPS:
                            KNOWN_COMBAT_MAPS[norm] = ("", canonical)
                count += 1

            elif cat in ("vật phẩm", "vat pham", "item"):
                for form in all_forms:
                    norm = normalize_game_text(form)
                    if norm:
                        CUSTOM_ITEM_ALIASES[norm] = canonical
                count += 1

        KNOWN_DIALOGUE_NPCS = tuple(known_npcs_list)
        return count
    except Exception as exc:
        logger.warning(f"Failed to load custom dictionary: {exc}")
        return 0


# Tự động nạp từ điển tùy chỉnh khi khởi động module
load_custom_dictionary()



def parse_yellow_entities(yellow_tokens: list[str]) -> dict:
    """
    Phân tích danh sách các cụm chữ màu vàng đọc được từ bảng nhiệm vụ thành các trường cụ thể:
    - count: số lượng cần (vd: 10)
    - map_name: tên bản đồ quái (vd: Tầng 2 Dương Trung động, Tường Vân động)
    - sect: môn phái / địa bàn (vd: Cái Bang, Ngũ Độc)
    - monster: tên quái (vd: Bát phụ Hà Đông Sư, Bán nương, Người lùn, Nhện độc)
    - item: vật phẩm cần giao nộp (vd: Trứng đại bàng, Lông đại bàng, Bảo tay)
    - city: thành phố cần đến (vd: Biện Kinh, Thành Đô)
    - npc: tên NPC cần gặp (vd: Khấu Chuẩn, Dã Tẩu, Chủ Tiệm Nữ phục)
    """
    import re
    from .text_recognition import normalize_game_text, fuzzy_score

    result = {
        "count": None,
        "map_name": "",
        "sect": "",
        "monster": "",
        "item": "",
        "city": "",
        "npc": "",
    }
    if not yellow_tokens:
        return result

    filtered_tokens = []
    for tok in yellow_tokens:
        t_clean = re.sub(r"['`\"^]", "", tok.strip())
        norm = normalize_game_text(t_clean)
        # Bỏ qua các nhãn header UI, thông tin chiếu chỉ hoặc giới hạn nhiệm vụ ngày
        if any(hdr in norm for hdr in (
            "thuong hoi", "tuan hoan", "nhiem vu", "danh sach", "noi dung",
            # Nhãn nhiệm vụ HUD bên trái, không phải thực thể vàng
            # trong nội dung nhiệm vụ.
            "huong dan tan thu", "huong dan tan thi", "tan thu",
            "20 lan", "lan nhiem vu", "toi da 20", "moi ngay", "ban chieu", "hoang thuong"
        )) or norm in ("lan", "20 lan", "20", "nhan toi da"):
            continue
        filtered_tokens.append(t_clean)

    # Kiểm tra xem đây có phải là nhiệm vụ đánh quái hay không
    is_combat_quest = False
    for tok in filtered_tokens:
        norm = normalize_game_text(tok)

        # Bỏ token OCR quá ngắn/rác (ví dụ "oi", "g1"). Những token này có
        # thể fuzzy-match nhầm một tên quái và ghi đè kết quả đúng trước đó.
        if len(norm.replace(" ", "")) < 3 and not norm.isdigit():
            continue
        sect_chk, map_chk = parse_sect_and_combat_map(tok, with_accents=False)
        is_known_monster_token = any(
            alias in norm
            or norm in alias
            or (
                bool(set(norm.split()) & set(alias.split()))
                and fuzzy_score(norm, alias) >= 0.78
            )
            for alias in KNOWN_MONSTER_ALIASES
        )
        # "đồng" và "tầng" đứng riêng xuất hiện rất nhiều trong chat/HUD
        # (vd. "đồng giá", "Cái Bang đường chủ"). Chỉ coi là dấu hiệu combat
        # khi parser nhận ra map/tên quái hoặc có từ chiến đấu rõ ràng.
        if map_chk or is_known_monster_token or any(w in norm for w in ("me cung", "quai", "diet", "trach", "coc doc", "goc dpc", "coc dpc", "goc doc", "bat phy", "ha dor", "ha dong su")):
            is_combat_quest = True
            break

    for tok in filtered_tokens:
        norm = normalize_game_text(tok)

        if len(norm.replace(" ", "")) < 3 and not norm.isdigit():
            continue

        # 1. Số lượng (bỏ qua nếu là số lần nhiệm vụ trong ngày như '20 lần')
        if "lan" in norm.split() or "lan" in norm:
            continue
        num_m = re.match(r"^(\d+)(?:\s+(?:con|cai|mon|vien|cay|bong|ten|chiec|binh))?$", norm)
        if num_m:
            result["count"] = int(num_m.group(1))
            continue

        # Nếu token bắt đầu bằng số (ví dụ: "10 Người lùn", "10 con Nhện độc", "10 tên Bát phụ")
        lead_num_m = re.match(r"^(\d+)\s*(?:con|cai|mon|vien|cay|bong|ten|chiec|binh|ke|a)?\s*(.+)$", norm)
        if lead_num_m and not norm.startswith(("tang ", "tang1", "tang2", "tang3", "tang4", "tang5")):
            val = int(lead_num_m.group(1))
            sub_norm = lead_num_m.group(2).strip()
            if 0 < val <= 500 and len(sub_norm) >= 3 and not any(bad in sub_norm for bad in ("th", "van quan", "quan lieu", "quoc gia", "chien su")):
                if result["count"] is None:
                    result["count"] = val
                norm = sub_norm

        # 2. Thành phố / Địa điểm (ưu tiên khớp thành phố trước)
        is_city = False
        norm_words = norm.split()
        for alias, c_name in SPECIAL_CITY_ALIASES.items():
            alias_words = alias.split()
            if norm == alias:
                result["city"] = c_name
                is_city = True
                break
            if (
                len(norm_words) <= len(alias_words) + 1
                and re.search(r"\b" + re.escape(alias) + r"\b", norm)
                and not any(w in norm_words for w in ("chu", "tiem", "npc", "quan", "ve", "binh", "tao", "khau", "pham", "luc"))
            ):
                result["city"] = c_name
                is_city = True
                break
        if not is_city:
            for alias, c_name in SPECIAL_CITY_ALIASES.items():
                alias_words = alias.split()
                if (
                    abs(len(norm_words) - len(alias_words)) <= 1
                    and fuzzy_score(norm, alias) >= 0.80
                    and not any(w in norm_words for w in ("chu", "tiem", "npc", "quan", "ve", "binh", "tao", "khau", "pham", "luc"))
                ):
                    result["city"] = c_name
                    is_city = True
                    break
        if is_city:
            continue

        # 3. Whitelist hàng Thương nhân Tây Vực (ưu tiên trước NPC để không bao giờ nhầm vật phẩm như 'bảo đao' thành 'Bộ đầu')
        matched_item = match_western_merchant_item(tok)
        if matched_item:
            result["item"] = matched_item
            continue

        # 4. Tên NPC đối thoại
        if any(bad in norm for bad in ("con dan", "clan dan", "dai tong", "quan lieu", "van quan", "quoc gia", "chien su")):
            continue

        is_npc = False
        norm_words = set(norm.split())
        has_nam = "nam" in norm_words
        has_nu = bool(norm_words & {"nu", "ni", "nt"})

        if norm in NPC_ALIASES:
            result["npc"] = NPC_ALIASES[norm]
            is_npc = True
        else:
            scored_candidates = []
            for alias, canonical in NPC_ALIASES.items():
                a_words = set(alias.split())
                c_words = set(normalize_game_text(canonical).split())
                if _is_shop_type_incompatible(norm_words, a_words) or _is_shop_type_incompatible(norm_words, c_words):
                    continue
                if has_nam and (c_words & {"nu", "ni", "nt"} or a_words & {"nu", "ni", "nt"}):
                    continue
                if has_nu and ("nam" in c_words or "nam" in a_words):
                    continue
                if alias in ("can quan", "cam quan") or canonical == "Cấm quân":
                    if not (norm_words & {"cam", "can", "kam"} and (norm_words & {"quan", "qyan", "qun"} or "quan" in norm)):
                        continue
                min_l = min(len(norm), len(alias))
                th = 0.85 if min_l <= 7 else 0.75
                sc = fuzzy_score(norm, alias)
                if alias == norm or (alias in norm and len(alias) >= 8) or sc >= th:
                    scored_candidates.append((sc, canonical))
            if scored_candidates:
                result["npc"] = max(scored_candidates, key=lambda x: x[0])[1]
                is_npc = True

        if not is_npc:
            has_refugee_sub = bool(norm_words & {"giap", "at", "binh", "dinh"})
            scored_known = []
            for known_npc in KNOWN_DIALOGUE_NPCS:
                norm_k = normalize_game_text(known_npc)
                k_words = set(norm_k.split())
                if _is_shop_type_incompatible(norm_words, k_words):
                    continue
                if has_nam and (k_words & {"nu", "ni", "nt"}):
                    continue
                if has_nu and ("nam" in k_words):
                    continue
                if known_npc == "Cấm quân" or norm_k in ("cam quan", "can quan"):
                    if not (norm_words & {"cam", "can", "kam"} and (norm_words & {"quan", "qyan", "qun"} or "quan" in norm)):
                        continue
                # Tránh nhầm lẫn "Nạn dân" với "Nạn dân Giáp/Ất/Bính/Đinh"
                k_has_sub = bool(k_words & {"giap", "at", "binh", "dinh"})
                if has_refugee_sub != k_has_sub and ("nan dan" in norm_k or "nan dan" in norm):
                    continue
                min_l = min(len(norm), len(norm_k))
                th = 0.85 if min_l <= 7 else 0.75
                sc = fuzzy_score(norm, norm_k)
                if norm == norm_k:
                    scored_known.append((2.0, known_npc))
                elif min_l >= 4 and sc >= th:
                    scored_known.append((sc, known_npc))
            if scored_known:
                result["npc"] = max(scored_known, key=lambda x: x[0])[1]
                is_npc = True
        if not is_npc:
            npc_titles = (
                "chu tiem", "chu tap hoa", "chu kim hoan", "chu tien trang",
                "chu tuu lau", "chu tiu lau", "cho ti lau", "tuu lau",
                "ve binh", "su gia", "bo dau", "dao nhan", "thuyen phu",
                "nan dan", "duong chu", "khau chuan", "pham trong yem",
                "vuong nghiep", "da tau", "bach hieu sinh", "quan thanh", "chi huy",
                "quan vien thoai nhiem", "quan vien thoai niem", "nuong", "dai phu",
                "thay boi", "thay", "cam quan", "chu tiem nam", "chu tiem nu",
                "nha dich", "nhadich", "nha dlch", "nha djch",
                "vu khi", "tiem vu khi", "chu tiem vu khi"
            )
            # OCR cắt đuôi tracker có thể tạo token như "g duong chu"; ký tự
            # rác một chữ ở đầu không phải là một tên NPC hợp lệ.
            malformed_prefix = bool(norm.split() and len(norm.split()[0]) == 1)
            if not malformed_prefix and any(title in norm for title in npc_titles):
                result["npc"] = get_canonical_npc_name(tok, original_text=tok)
                is_npc = True
        if is_npc:
            continue

        # 3.5. Kiểm tra map quái đã biết trong KNOWN_COMBAT_MAPS trước tiên để tránh bị nhận nhầm
        # sang tên quái có phát âm tương tự (ví dụ: 'Miêu Lĩnh' bị nhận nhầm sang 'Liêu binh')
        if norm in KNOWN_COMBAT_MAPS and not result["map_name"]:
            sect, c_map = KNOWN_COMBAT_MAPS[norm]
            result["map_name"] = c_map
            if sect:
                result["sect"] = sect
            continue

        # Tên quái phải được xét trước tên map thông thường. Một số tên riêng như
        # "Bát phụ Hà Đông Sư" kết thúc bằng chữ "Đông" và trước đây bị hiểu
        # nhầm thành một bản đồ có hậu tố "động".
        if not result["monster"]:
            for alias, canon in KNOWN_MONSTER_ALIASES.items():
                if canon == "Bán nương" and not norm.startswith(("ban", "barnh")):
                    continue
                if alias in norm or norm in alias or fuzzy_score(norm, alias) >= 0.70:
                    result["monster"] = canon
                    break
            if result["monster"]:
                continue

        # 4. Bản đồ quái (tách rõ địa bàn môn phái và map quái, loại trừ thành lớn)
        sect, c_map = parse_sect_and_combat_map(tok, with_accents=True)
        if c_map and not result["map_name"]:
            result["map_name"] = c_map
            if sect:
                result["sect"] = sect
            continue

        # 5. Tên quái từ danh mục đã biết KNOWN_MONSTER_ALIASES
        is_known_monster = False
        if not result["monster"]:
            for alias, canon in KNOWN_MONSTER_ALIASES.items():
                if alias in norm or norm in alias or fuzzy_score(norm, alias) >= 0.70:
                    result["monster"] = canon
                    is_known_monster = True
                    break
        if is_known_monster:
            continue

        # 6. Vật phẩm shop: chỉ chấp nhận tên có trong whitelist Thương nhân
        # Tây Vực. Không dùng từ chung như "kiếm/đá/ngọc/dược" vì đó có thể
        # là tên quái, NPC hoặc địa danh.
        if not is_combat_quest:
            matched_item = match_western_merchant_item(tok)
            if matched_item:
                result["item"] = matched_item
                continue
            for a_norm, canon_item in CUSTOM_ITEM_ALIASES.items():
                if norm == a_norm or (len(a_norm) >= 6 and a_norm in norm) or fuzzy_score(norm, a_norm) >= 0.75:
                    result["item"] = canon_item
                    break
            if result["item"]:
                continue

        # 7. Tên quái theo từ khóa đặc trưng
        is_map_token = any(kw in norm for kw in ("me cung", "tang", "dong", "son", "thon", "rung", "mo", "trai", "trach"))
        if not is_map_token and (norm == "coc" or norm.endswith(" coc")):
            if not any(m in norm for m in ("coc doc", "quai", "thu")):
                is_map_token = True
        if is_map_token:
            continue

        monster_keywords = (
            "quai", "thu", "tac", "soi", "ho", "chuot", "binh", "nguu", "ma",
            "yeu", "tru", "xa", "dao tac", "nhen", "doc", "coc", "bo cap", "ong",
            "sau", "ret", "gau", "heo", "cao", "ran", "phi tac", "hai tac",
            "tho phi", "gian te", "lieu quan", "quan lieu", "su", "dor", "phy", "phu",
            "nuong", "lun", "phan tan", "tao phan",
            "thao", "nhan thao", "thuc nhan thao", "thuc nhan", "thuc vat"
        )
        norm_words = set(norm.split())
        is_monster = False
        for kw in monster_keywords:
            if kw == "doc" and any(nd in norm for nd in ("ngu doc", "ngi doc", "doc giao")):
                continue
            if (" " not in kw and kw in norm_words) or (" " in kw and kw in norm):
                is_monster = True
                break

        if is_monster and not result["monster"]:
            canonical_m = None
            for alias, canon in KNOWN_MONSTER_ALIASES.items():
                if alias in norm or (len(norm) >= 4 and norm in alias) or fuzzy_score(norm, alias) >= 0.70:
                    canonical_m = canon
                    break
            result["monster"] = canonical_m or tok
            continue

        # Không đoán mọi token vàng chưa biết là vật phẩm. Tên NPC mới/chưa có
        # trong danh mục cũng là token tự do và trước đây bị nhận nhầm thành đồ mua.

    # Nếu là nhiệm vụ đánh quái nhưng chưa trích xuất được quái:
    # Tìm xem có token nào còn sót lại chưa gán (không phải số, không phải map)
    if is_combat_quest and not result["monster"]:
        for tok in filtered_tokens:
            norm = normalize_game_text(tok)
            if len(norm.replace(" ", "")) < 3 and not norm.isdigit():
                continue
            if norm.isdigit():
                continue
            # Không tái sử dụng tên NPC hoặc chữ giao diện/chat làm tên quái.
            npc_noise = (
                "duong chu", "chu thuong hoi", "chu tiem", "chu tap hoa",
                "khau chuan", "pham trong yem", "vuong nghiep", "da tau",
                "tong hop", "he thong", "mon phai", "bang", "nhiem vu",
                "huong dan tan thu", "dong gia", "gia re",
            )
            if any(noise in norm for noise in npc_noise):
                continue
            if any(kw in norm for kw in ("tang", "dong", "son", "thon", "rung", "mo", "me cung", "trach")):
                continue
            if norm == "coc" or norm.endswith(" coc"):
                if not any(m in norm for m in ("coc doc", "quai", "thu")):
                    continue
            canonical_m = None
            for alias, canon in KNOWN_MONSTER_ALIASES.items():
                if alias in norm or norm in alias or fuzzy_score(norm, alias) >= 0.70:
                    canonical_m = canon
                    break
            result["monster"] = canonical_m or tok
            break

    return result


def extract_shop_item(text: str, yellow_tokens: Optional[list[str]] = None) -> str:
    """Extract the requested item from Western Merchant quest wording or yellow tokens."""
    import re

    if yellow_tokens:
        entities = parse_yellow_entities(yellow_tokens)
        if entities.get("item"):
            return entities["item"]

    matched_item = match_western_merchant_item(text)
    if matched_item:
        return matched_item

    value = normalize_game_text(text)
    patterns = (
        r"dat (?:lam |mua )?(?:mot |[0-9]+\s+)?(?:binh |chiec |cai |vien |cuon |tam |mon |hop |bao |goi |soi |bo |mieng |khoi |can |lang )?(.+?)(?:\s+[0o]\s*cho\s+thuong\s*nhan|\s+o cho thuong nhan|\s+[0o]\s+thuong\s*nhan|\s+o thuong nhan|\s+tai thuong nhan|\s+tu thuong nhan|\s+[0o]\s*cho|\s+o cho|\s+tai|\s+tu|\s+xin giup)",
        r"(?:mua|lay|tim mua|dat)\s+(?:mot |[0-9]+\s+)?(?:binh |chiec |cai |vien |cuon |tam |mon |hop |bao |goi |soi |bo |mieng |khoi |can |lang )?([a-z0-9 ]+?)(?:\s+[0o]\s*cho|\s+o cho|\s+[0o]\s+thuong\s*nhan|\s+o thuong nhan|\s+tu|\s+tai|\s+ve|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            item = match.group(1).strip()
            item = re.sub(r"^(?:lam\s+|mua\s+)?(?:\d+\s+)?(?:mot\s+|binh\s+|chiec\s+|cai\s+|vien\s+|cuon\s+|tam\s+|mon\s+|hop\s+|bao\s+|goi\s+|soi\s+|bo\s+|mieng\s+|khoi\s+|can\s+|lang\s+)?", "", item).strip()
            item = clean_shop_item_name(item)
            matched = match_western_merchant_item(item)
            if matched:
                return matched

    # Fallback trực tiếp danh mục vật phẩm Thương nhân Tây Vực đã biết
    KNOWN_WESTERN_ITEMS = (
        ("linh duoc tay vuc", "Linh dược Tây Vực"),
        ("linh duoc", "Linh dược Tây Vực"),
        ("long dai bang", "Lông đại bàng"),
        ("trung dai bang", "Trứng đại bàng"),
        ("thit lac da", "Thịt lạc đà"),
        ("ho uyen long lac da", "Hộ uyển lông lạc đà"),
        ("ho uyen lac da", "Hộ uyển lạc đà"),
        ("ho uyen", "Hộ uyển"),
        ("long lac da", "Lông Lạc Đà"),
        ("long lac", "Lông Lạc Đà"),
        ("bao tay tay vuc", "Bảo tay Tây Vực"),
        ("bao tay", "Bảo tay"),
        ("tinh kim tay vuc", "Tinh kim Tây Vực"),
        ("tinh kim", "Tinh kim"),
        ("kim cang nguyen thach", "Kim cang nguyên thạch"),
        ("tuyet tinh nguyen thach", "Tuyết tinh nguyên thạch"),
        ("am da nguyen thach", "Ám dạ nguyên thạch"),
        ("am vien nguyen thach", "Ám viên nguyên thạch"),
        ("da quang boi", "Dạ quang bôi"),
    )
    for key, name in KNOWN_WESTERN_ITEMS:
        if key in value:
            return name

    return ""


def extract_combat_target(text: str, yellow_tokens: Optional[list[str]] = None) -> tuple[str, str, int]:
    """
    Extract map, monster name, and required count from combat quest wording or yellow tokens.
    Hỗ trợ cả token chữ vàng và các mẫu câu nhiệm vụ đánh quái trong game.
    Tách biệt rõ ràng địa bàn môn phái (ví dụ Cái Bang, Ngũ Độc) và tên map quái (Tường Vân động, Ngũ Độc Mê Cung tầng 2).
    Loại trừ tuyệt đối tên các thành lớn (Biện Kinh, Thành Đô, trong thành...) để tránh hiểu lầm.
    """
    import re

    c_map = ""
    c_monster = ""
    c_count = 0

    # Tiền xử lý: Cắt bỏ các đoạn hội thoại lựa chọn, câu hỏi đưa đi hoặc nhiệm vụ phụ trên HUD
    clean_text = text or ""
    clean_text = re.split(
        r"(?:co\s+can\s+ta\s+dua|dua\s+den\s+noi\s+lam|phien\s+dua\s+ta|khong\s+can\s+dau|cung\s+voi\s+bach\s+hieu\s+sinh|nhiem\s+vu\s+mon\s+phai)",
        clean_text,
        flags=re.IGNORECASE,
    )[0]
    clean_text = re.sub(
        r"^(?:y\s+)?huong\s+dan\s+tan\s+thu\s+(?:auto\s+)?(?:ie\s+)?",
        "",
        clean_text,
        flags=re.IGNORECASE,
    )

    normalized = normalize_game_text(clean_text)
    explicit_dialogue = any(kw in normalized for kw in ("noi voi", "doi thoai voi", "thong bao cho", "bao tin cho", "nhan tin cho", "gap", "hoi tham", "chuyen thu"))
    has_combat_action = any(kw in normalized for kw in ("diet", "giet", "trung tri", "truing tri", "tring tri", "danh bai", "ha guc", "tieu diet", "bat"))
    if explicit_dialogue and not has_combat_action:
        return "", "", 0

    # Tiền xử lý: OCR thường đọc chữ 'ở' trong game thành số '0' hoặc chữ 'o'
    norm_prep = re.sub(r"\b0\b", "o", normalized)
    norm_prep = re.sub(r"^(?:chu\s+thuong\s+hoi|thuong\s+hoi|nhiem\s+vu)\s*", "", norm_prep)

    for g, r in [
        ('xungquanh', 'xung quanh '), ('vanmong', 'van mong '),
        ('trachc6nhieu', 'trach co nhieu '), ('trach6nhieu', 'trach co nhieu '),
        ('c6nhieu', ' co nhieu '), ('c6', ' co '),
        ('trachconhieu', 'trach co nhieu '), ('conhieu', ' co nhieu '),
        ('cocdoc', 'coc doc '), ('docgayhai', 'doc gay hai '),
        ('gayhai', ' gay hai '), ('haydidiet', ' hay di diet '),
        ('didiet', ' di diet '),
    ]:
        norm_prep = norm_prep.replace(g, r)

    # Tách từ dính liền với số
    norm_prep = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', norm_prep)
    norm_prep = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', norm_prep)
    norm_prep = ' '.join(norm_prep.split())

    INVALID_COMBAT_MAP_PHRASES = (
        "den noi lam", "lam nhiem vu", "nhiem vu", "co can", "phien dua",
        "khong can", "chuyen thu", "doi thoai", "bach hieu sinh", "tan thu"
    )

    def _sanitize_combat_map(raw_m: str) -> str:
        if not raw_m:
            return ""
        norm_m = normalize_game_text(raw_m)
        if any(p in norm_m for p in INVALID_COMBAT_MAP_PHRASES):
            return ""
        for alias, canon in KNOWN_SECTS.items():
            if alias in norm_m or norm_m in alias:
                return canon
        return raw_m

    # 1. Ưu tiên trích xuất từ các token chữ vàng nếu có
    if yellow_tokens:
        entities = parse_yellow_entities(yellow_tokens)
        c_map = _sanitize_combat_map(entities.get("map_name", ""))
        c_monster = entities.get("monster", "")
        c_count = entities.get("count", 0) or 0
        if c_map and c_monster and c_count > 0:
            return c_map, c_monster, c_count
        if c_map and c_monster and c_count == 0:
            # Tìm số lượng từ text mô tả
            m_cnt = re.search(r"\b([1-9]\d*)\b", norm_prep)
            return c_map, c_monster, int(m_cnt.group(1)) if m_cnt else 10

    # Mẫu hội thoại Chủ thương hội: tên quái đứng trước địa điểm, còn số lượng
    # nằm sau động từ "trừng trị". Ví dụ thực tế:
    # "có rất nhiều Nữ Kiếm Khách ... tụ tập ở Nga My Phái gây rối,
    #  hãy đi trừng trị 10 ả".
    chief_match = re.search(
        r"co\s+(?:rat\s+)?nhieu\s+([a-z0-9 ]{2,40}?)\s+"
        r"(?:bi\s+.+?\s+)?(?:t[uy]\s*tap|tap\s*trung|tu\s*hop|hoat\s*dong|xuat\s*hien)\s+(?:o|tai|0)\s+"
        r"([a-z0-9 ]{2,30}?)\s+"
        r"(?:gay\s+roi|quay\s+pha|gay\s+hai|lam\s+loan|gay\s+chuyen|lam\s+nhieu|lam\s+phan|tao\s+phan|gay\s+kho)?"
        r"[^,!.0-9]*?"
        r"(?:hay\s+)?(?:di\s+)?(?:tr[ui]ng\s+tri|trinh\s+tri|truing\s+tri|tring\s+tri|diet|tieu\s+diet|giet|danh\s+bai|ha\s+guc)\s+"
        r"(\d+)\s*(?:con|ten|ke|a|chiec|vien)?\b",
        norm_prep,
    )
    if chief_match:
        raw_monster = chief_match.group(1).strip()
        raw_location = chief_match.group(2).strip()
        count = int(chief_match.group(3))

        monster = raw_monster.title()
        for alias, canon in KNOWN_MONSTER_ALIASES.items():
            if alias in raw_monster or raw_monster in alias or fuzzy_score(raw_monster, alias) >= 0.70:
                monster = canon
                break

        location = raw_location.title()
        for alias, canon in KNOWN_SECTS.items():
            if alias in raw_location or raw_location in alias or fuzzy_score(raw_location, alias) >= 0.78:
                location = canon
                break
        return location, monster, count

    # Mẫu hội thoại Chủ thương hội 2: Địa điểm đứng trước "có rất nhiều [quái] làm phản/gây rối..."
    # Ví dụ: "Luong Son Bac co rat nhieu Son tac lam phan, hay di truing tri 10 ten"
    chief_match_loc_first = re.search(
        r"([a-z0-9 ]{2,60}?)\s+co\s+(?:rat\s+)?nhieu\s+([a-z0-9 ]{2,40}?)\s+"
        r"(?:lam\s+phan|tao\s+phan|gay\s+roi|quay\s+pha|gay\s+hai|tu\s+tap|lam\s+loan|quay\s+nhieu)[,!.]?\s*.*?"
        r"(?:tr[ui]ng\s+tri|trinh\s+tri|truing\s+tri|tring\s+tri|diet|tieu\s+diet|giet|danh\s+bai|ha\s+guc)\s+"
        r"(\d+)\s*(?:con|ten|ke|a)?\b",
        norm_prep,
    )
    if chief_match_loc_first and chief_match_loc_first.group(2).strip() not in ("ke", "ten", "nguoi", "chung", "chung no", "thon dan"):
        raw_location = chief_match_loc_first.group(1).strip()
        raw_monster = chief_match_loc_first.group(2).strip()
        count = int(chief_match_loc_first.group(3))

        monster = raw_monster.title()
        for alias, canon in KNOWN_MONSTER_ALIASES.items():
            if alias in raw_monster or raw_monster in alias or fuzzy_score(raw_monster, alias) >= 0.70:
                monster = canon
                break

        location = raw_location.title()
        _, clean_m = parse_sect_and_combat_map(raw_location, with_accents=True)
        if clean_m:
            location = clean_m
        else:
            for alias, canon in KNOWN_SECTS.items():
                if alias in raw_location or raw_location in alias or fuzzy_score(raw_location, alias) >= 0.78:
                    location = canon
                    break
        return location, monster, count

    # Mẫu 0A (Đặc trưng ưu tiên: "hãy đi diệt [số] tên/con [Tên quái]"):
    # "Tầng 2 Dương Trung động có rất nhiều kẻ tụ tập tạo phản, hãy đi diệt 10 tên Bát phụ Hà Đông Sư"
    match0a = re.search(
        r"(?:hay\s+)?(?:di\s+)?(?:diet|tieu\s+diet|giet|danh\s+bai|ha\s+guc|trung\s+tri)\s+(\d+)\s+(?:con|cay|bong|ten|chiec|vien|ke)\s+([a-z0-9 ]{2,35}?)(?:\.|$|,|\s+co\s+can|\s+dem\s+ve)",
        norm_prep
    )
    if match0a:
        m_count = int(match0a.group(1).strip())
        raw_monster = match0a.group(2).strip()
        if raw_monster not in ("con", "cay", "bong", "ten", "chiec", "vien", "ke", ""):
            canonical_monster = None
            for alias, canon in KNOWN_MONSTER_ALIASES.items():
                if alias in raw_monster or (len(raw_monster) >= 5 and raw_monster in alias) or fuzzy_score(raw_monster, alias) >= 0.70:
                    canonical_monster = canon
                    break

            is_valid_monster = bool(canonical_monster or any(kw in raw_monster.split() for kw in ("quai", "thu", "tac", "soi", "ho", "chuot", "binh", "nguu", "ma", "yeu", "tru", "xa", "nhen", "doc", "coc", "su", "dor", "phy", "phu", "nuong", "lun", "thao")))
            if is_valid_monster:
                if yellow_tokens:
                    final_monster = canonical_monster or raw_monster.title()
                    final_map = c_map
                    if not final_map:
                        map_m = re.search(r"\b([a-z0-9 ]{2,35}?\b(?:dong|son|quang|dao|coc|thon|mo|rung|ho|trai|trach|quan|me\s+cung(?:\s+tang\s+\d+|\s+\d+)?))\b", norm_prep)
                        if map_m:
                            _, clean_map = parse_sect_and_combat_map(map_m.group(1).strip(), with_accents=True)
                            final_map = clean_map or map_m.group(1).strip()
                    return final_map or c_map, c_monster or final_monster, c_count or m_count
                else:
                    final_monster = raw_monster
                    map_before = re.search(r"([a-z0-9 ]{2,35}?(?:dong|son|quang|dao|coc|thon|mo|rung|ho|trai|trach|quan|me\s+cung(?:\s+tang\s+\d+|\s+\d+)?))\s+co\s+(?:rat\s+)?nhieu", norm_prep)
                    clean_map = ""
                    if map_before:
                        raw_m = map_before.group(1).strip()
                        _, c_m = parse_sect_and_combat_map(raw_m, with_accents=False)
                        clean_map = c_m or raw_m
                    else:
                        map_m = re.search(r"(?:(?:xung\s+quanh|sau\s+nui\s+[a-z0-9 ]+?\s+)?(?:o|tai|den|toi\s+)?)\s*([a-z0-9 ]{2,35}?(?:dong|son|quang|dao|coc|thon|mo|rung|ho|trai|trach|quan|me\s+cung(?:\s+tang\s+\d+|\s+\d+)?))", norm_prep)
                        if map_m:
                            raw_m = map_m.group(1).strip()
                            raw_m = re.sub(r"^(?:hay\s+den|den|toi|o|tai|xung\s+quanh)\s+", "", raw_m).strip()
                            _, c_m = parse_sect_and_combat_map(raw_m, with_accents=False)
                            clean_map = c_m or raw_m
                    return clean_map, final_monster, m_count

    # Mẫu 0 (Nhiệm vụ nguy hiểm / Thực nhân thảo / Mê cung):
    # "[Map] có loài [Quái] rất nguy hiểm, hãy đi diệt trừ [Số] cây/con"
    match0 = re.search(
        r"(?:(?:xung\s+quanh|sau\s+nui\s+[a-z0-9 ]+?\s+)?(?:o|tai|den|toi\s+)?)\s*([a-z0-9 ]{2,45}?(?:dong|son|quang|dao|coc|thon|mo|rung|ho|trai|trach|quan|me\s+cung(?:\s+tang\s+\d+|\s+\d+)?))\s+co\s+(?:rat\s+nhieu|loai)\s+([a-z0-9 ]+?)(?:,|\.|\s+rat\s+nguy\s+hiem|\s+khien|\s+hay|\s+lam)\b.*?(?:hay\s+)?(?:di\s+)?(?:diet\s+tri|diet|tieu diet|giet|danh bai|ha guc)\s+(\d+)\s*(?:con|cay|bong|ten|chiec|vien)?",
        norm_prep
    )
    if match0:
        raw_map = match0.group(1).strip()
        _, clean_map = parse_sect_and_combat_map(raw_map, with_accents=False)
        map_name = clean_map or raw_map
        monster = match0.group(2).strip()
        count = int(match0.group(3).strip())
        return map_name, monster, count

    # Mẫu 1 (Mẫu thực tế trong game):
    # "[Địa bàn/Môn phái] [Map quái] có rất nhiều [tên quái]... hãy đi diệt/bắt [số] con!"
    # Ví dụ: "Cái Bang Tường Vân động có rất nhiều nhện độc, hãy đi diệt 10 con!"
    # "Gần Vân Mộng trạch có rất nhiều cóc độc, hãy đi bắt 10 con."
    match1 = re.search(
        r"(?:(?:xung\s+quanh|sau\s+nui\s+[a-z0-9 ]+?\s+|gan\s+)?(?:o|tai|den|toi\s+)?)\s*([a-z0-9 ]{2,35}?(?:dong|son|quang|dao|coc|thon|mo|rung|ho|trai|trach|quan))\s+co\s+(?:rat\s+)?nhieu\s+([a-z0-9 ]+?)(?:,|\.|\s+gay\s+hai|\s+quay\s+pha|\s+khien|\s+hay|\s+lam)\b.*?(?:hay\s+)?(?:di\s+)?(?:diet|tieu\s+diet|giet|danh\s+bai|ha\s+guc|bat|trung\s+tri)\s+(\d+)\s*(?:con|cay|bong|ten)?",
        norm_prep
    )
    if match1:
        raw_map = match1.group(1).strip()
        _, clean_map = parse_sect_and_combat_map(raw_map, with_accents=False)
        map_name = clean_map or raw_map
        monster = match1.group(2).strip()
        count = int(match1.group(3).strip())
        canonical_monster = None
        for alias, canon in KNOWN_MONSTER_ALIASES.items():
            if alias in monster or monster in alias or fuzzy_score(monster, alias) >= 0.70:
                canonical_monster = canon
                break
        return map_name, canonical_monster or monster, count

    # Mẫu 1B (Xung quanh/Gần [Map] có nhiều [Quái] gây hại... hãy đi diệt/bắt [Số] con):
    # Ví dụ: "Xung quanh Vân Mộng trạch có nhiều Cóc độc gây hại, hãy đi diệt 10 con ."
    match1b = re.search(
        r"(?:(?:xung\s+quanh|sau\s+nui\s+[a-z0-9 ]+?\s+|gan\s+)?(?:o|tai|den|toi\s+)?)\s*([a-z0-9 ]{2,35}?(?:dong|son|quang|dao|coc|thon|mo|rung|ho|trai|trach|quan|me\s+cung(?:\s+tang\s+\d+|\s+\d+)?))\s+co\s+(?:rat\s+)?nhieu\s+([a-z0-9 ]+?)(?:,|\.|\s+gay\s+hai|\s+quay\s+pha|\s+khien|\s+hay|\s+lam)\b.*?(?:hay\s+)?(?:di\s+)?(?:diet|tieu\s+diet|giet|danh\s+bai|ha\s+guc|bat|trung\s+tri)\s+(\d+)\s*(?:con|cay|bong|ten)?",
        norm_prep
    )
    if match1b:
        raw_map = match1b.group(1).strip()
        _, clean_map = parse_sect_and_combat_map(raw_map, with_accents=False)
        map_name = clean_map or raw_map
        monster = match1b.group(2).strip()
        count = int(match1b.group(3).strip())
        canonical_monster = None
        for alias, canon in KNOWN_MONSTER_ALIASES.items():
            if alias in monster or monster in alias or fuzzy_score(monster, alias) >= 0.70:
                canonical_monster = canon
                break
        return map_name, canonical_monster or monster, count

    # Mẫu 2: "Hãy đến Tường Vân động tiêu diệt 20 Sói điên"
    match2 = re.search(
        r"(?:hay den|den|toi)\s+([a-z0-9 ]+?)\s+(?:tieu diet|danh bai|ha guc|giet)\s+(\d+)\s+([a-z0-9 ]+?)(?:\.|$|,|\s+dem|\s+ve)",
        norm_prep
    )
    if match2:
        raw_map = match2.group(1).strip()
        _, clean_map = parse_sect_and_combat_map(raw_map, with_accents=False)
        map_name = clean_map or raw_map
        count = int(match2.group(2).strip())
        monster = match2.group(3).strip()
        return map_name, monster, count

    # Mẫu 3: "tiêu diệt 10 tuyết quái tại Băng Tâm động"
    match3 = re.search(
        r"(?:tieu diet|danh bai|ha guc|giet|diet)\s+(\d+)\s+([a-z0-9 ]+?)\s+(?:o|tai|den|trong)\s+([a-z0-9 ]+?)(?:\.|$|,)",
        norm_prep
    )
    if match3:
        count = int(match3.group(1).strip())
        monster = match3.group(2).strip()
        raw_map = match3.group(3).strip()
        _, clean_map = parse_sect_and_combat_map(raw_map, with_accents=False)
        map_name = clean_map or raw_map
        return map_name, monster, count

    # Mẫu 4: Mẫu tổng quát khi có "có rất nhiều [quái] ... diệt [số] con"
    # Tuyệt đối loại trừ 'thanh', 'phu' để không bắt nhầm thành lớn / 'trong thành'
    match4 = re.search(
        r"co\s+(?:rat\s+)?nhieu\s+([a-z0-9 ]+?)(?:,|\.|\s+gay\s+hai|\s+khien|\s+hay|\s+lam)\b.*?(?:hay\s+)?(?:di\s+)?(?:diet|tieu diet|giet|danh bai|ha guc)\s+(\d+)\s+(?:con|cay|bong|ten)",
        norm_prep
    )
    if match4:
        monster = match4.group(1).strip()
        for alias, canon in KNOWN_MONSTER_ALIASES.items():
            if alias in monster or monster in alias or fuzzy_score(monster, alias) >= 0.70:
                monster = canon
                break
        count = int(match4.group(2).strip())
        map_match = re.search(r"\b([a-z0-9 ]{2,20}?(?:dong|son|quang|dao|coc|thon|mo|rung|ho|trai|me\s+cung(?:\s+tang\s+\d+|\s+\d+)?))\b", norm_prep)
        if map_match:
            raw_map = map_match.group(1).strip()
            _, clean_map = parse_sect_and_combat_map(raw_map, with_accents=False)
            map_name = clean_map or raw_map
        else:
            map_name = ""
        return map_name, monster, count
    # Mẫu 5 (Trừng trị thôn dân tạo phản / quái vật cụ thể theo sau số lượng):
    # "[Tầng 2 Dương Trung động] ở [Hạnh Hoa thôn] có nhiều thôn dân tạo phản, hãy đi trừng trị 10 tên Bát Phụ Hà Đông sư..."
    match5 = re.search(
        r"(?:(?:sau\s+nui\s+[a-z0-9 ]+?\s+)?(?:o|tai|den|toi\s+)?)\s*([a-z0-9 ]{2,45}?\b(?:dong|son|quang|dao|coc|thon|mo|rung|ho|trai|me\s+cung(?:\s+tang\s+\d+|\s+\d+)?))\b.*?(?:hay\s+)?(?:di\s+)?(?:tr[ui]ng\s+tri|trinh\s+tri|truing\s+tri|tring\s+tri|diet|tieu\s+diet|giet|danh\s+bai|ha\s+guc)\s+(\d+)\s*(?:con|cay|bong|ten|chiec|vien)?\s+([a-z0-9 ]+?)(?:\.|$|,|\s+co\s+can|\s+dem\s+ve)",
        norm_prep
    )
    if match5:
        raw_map = match5.group(1).strip()
        _, clean_map = parse_sect_and_combat_map(raw_map, with_accents=False)
        map_name = _sanitize_combat_map(clean_map or raw_map)
        count = int(match5.group(2).strip())
        monster = match5.group(3).strip()
        return map_name, monster, count

    # Mẫu 6 (Trừng trị / tiêu diệt không có tiền tố map rõ ràng):
    match6 = re.search(
        r"(?:hay\s+)?(?:di\s+)?(?:tr[ui]ng\s+tri|trinh\s+tri|truing\s+tri|tring\s+tri|diet|tieu\s+diet|giet|danh\s+bai|ha\s+guc)\s+(\d+)\s*(?:con|cay|bong|ten|chiec|vien)?\s+([a-z0-9 ]+?)(?:\.|$|,|\s+co\s+can|\s+dem\s+ve)",
        norm_prep
    )
    if match6:
        count = int(match6.group(1).strip())
        monster = match6.group(2).strip()
        map_match = re.search(r"\b([a-z0-9 ]{2,35}?\b(?:dong|son|quang|dao|coc|thon|mo|rung|ho|trai|me\s+cung(?:\s+tang\s+\d+|\s+\d+)?))\b", norm_prep)
        if map_match:
            raw_map = map_match.group(1).strip()
            _, clean_map = parse_sect_and_combat_map(raw_map, with_accents=False)
            map_name = _sanitize_combat_map(clean_map or raw_map)
        else:
            map_name = ""
        return map_name, monster, count

    final_map = _sanitize_combat_map(c_map)
    final_monster = c_monster
    if not final_map:
        for alias, canon in KNOWN_SECTS.items():
            if alias in norm_prep:
                final_map = canon
                break
    if not final_map:
        for k_map, (s_name, m_name) in KNOWN_COMBAT_MAPS.items():
            if k_map in norm_prep:
                final_map = m_name
                break
    if not final_monster:
        for alias, canon in KNOWN_MONSTER_ALIASES.items():
            if alias in norm_prep:
                final_monster = canon
                break

    # Fallback pattern tìm số lượng
    count_match = re.search(r"\b([1-9]\d*)\b", norm_prep)
    count = int(count_match.group(1)) if count_match else (c_count or (10 if (final_map and final_monster) else 0))
    return final_map, final_monster, count


class QuestClassifier:
    """

    Phân loại loại nhiệm vụ hiện tại.
    Không dùng ML ở Phase 1 – chỉ dùng template + icon.
    """

    # Mapping template name → QuestType
    ICON_TEMPLATE_MAP = {
        "quest_icon_npc":    QuestType.NPC,
        "quest_icon_shop":   QuestType.SPECIAL_NPC_SHOP,
        "quest_icon_combat": QuestType.COMBAT,
    }

    TEXT_RULES = {
        QuestType.SPECIAL_NPC_SHOP: (
            "thương nhân phương tây", "thương nhân tây vực", "thương nhân tây hồ",
            "mua vật phẩm", "mua đồ", "tìm mua", "linh dược tây vực", "đặt một bình",
            "lông đại bàng", "trứng đại bàng", "thịt lạc đà", "hộ uyển", "bảo tay",
        ),
        QuestType.COMBAT: (
            "đánh quái", "tiêu diệt", "đánh bại", "giết", "số lượng", "trừng trị", "tạo phản", "hà đông sư",
            "bắt", "đi bắt", "cóc độc", "vân mộng trạch",
        ),
        QuestType.NPC: (
            "đối thoại với", "nói chuyện", "gặp", "tìm npc", "hỏi thăm", "báo tin",
        ),
    }

    def __init__(self, vision: VisionEngine, config: AppConfig):
        self._vision = vision
        self.config = config

    def classify(self, frame: Optional[np.ndarray] = None, text: str = "", yellow_tokens: Optional[list[str]] = None) -> QuestType:
        """
        Phân loại quest type từ frame hiện tại, quest text, hoặc danh sách chữ vàng.
        Priority: yellow entities → text classification (với validation nghiêm ngặt) → icon template → UNKNOWN.
        """
        clean_text = text or ""
        clean_text = re.split(
            r"(?:co\s+can\s+ta\s+dua|dua\s+den\s+noi\s+lam|phien\s+dua\s+ta|khong\s+can\s+dau|cung\s+voi\s+bach\s+hieu\s+sinh|nhiem\s+vu\s+mon\s+phai)",
            clean_text,
            flags=re.IGNORECASE,
        )[0]
        clean_text = re.sub(
            r"^(?:y\s+)?huong\s+dan\s+tan\s+thu\s+(?:auto\s+)?(?:ie\s+)?",
            "",
            clean_text,
            flags=re.IGNORECASE,
        )
        norm_text_for_priority = normalize_game_text(clean_text) if clean_text else ""
        explicit_combat_text = any(kw in norm_text_for_priority for kw in (
            "tieu diet", "danh bai", "ha guc", "giet", "danh quai",
            "di diet", "trung tri", "tao phan", "gay roi", "di bat", "bat", "coc doc",
        ))
        has_active_npc_turnin = (
            "cung voi" in norm_text_for_priority and any(
                k in norm_text_for_priority for k in ("doi thoai", "hoan thanh", "hoan thanhnhiem vu")
            )
        ) or (
            "doi thoai" in norm_text_for_priority and "hoan thanh nhiem vu" in norm_text_for_priority
        )
        explicit_dialogue_text = any(kw in norm_text_for_priority for kw in (
            "noi voi", "noi chuyen voi", "doi thoai voi", "thong bao cho",
            "bao tin cho", "bao cho", "gap xa phu", "tim xa phu",
        )) or has_active_npc_turnin
        city_count_in_text = sum(
            1 for c in ("bien kinh", "thanh do", "tuyen chau", "tuong duong", "duong chau", "dai ly", "phuong tuong")
            if c in norm_text_for_priority
        )
        has_merchant_item = bool(
            match_western_merchant_item(norm_text_for_priority)
            or (yellow_tokens and parse_yellow_entities(yellow_tokens).get("item"))
            or extract_shop_item(text or "", yellow_tokens=yellow_tokens)
        )
        is_western_merchant_text = has_merchant_item and (
            "thuong nhan tay vuc" in norm_text_for_priority
            or "thuong nhan phuong tay" in norm_text_for_priority
            or "de tim ong ta" in norm_text_for_priority
            or "o cho thuong nhan" in norm_text_for_priority
            or "xin giup ta lay ve" in norm_text_for_priority
            or (("thuong nhan" in norm_text_for_priority or "tay vuc" in norm_text_for_priority) and any(w in norm_text_for_priority for w in ("dat", "mua", "lay", "de tim")))
            or city_count_in_text >= 4
        )

        if is_western_merchant_text:
            logger.info("Quest classified as SPECIAL_NPC_SHOP via Western Merchant text pattern")
            return QuestType.SPECIAL_NPC_SHOP

        # Mục tiêu đối thoại / trả nhiệm vụ ("cùng với ... đối thoại, hoàn thành nhiệm vụ")
        # là hành động trực tiếp quyết định loại nhiệm vụ NPC, ghi đè toàn bộ chữ quái trong mô tả.
        if has_active_npc_turnin:
            logger.info("Quest classified as NPC because active dialogue objective ('cùng với ... đối thoại') takes precedence")
            return QuestType.NPC

        # Câu lệnh cuối cùng "đến ... nói với ..." quyết định loại
        # nhiệm vụ. Không để các cụm nền như "10 vạn quân Liêu"
        # hoặc OCR vàng nhiễu ghi đè thành nhiệm vụ đánh quái.
        if explicit_dialogue_text and not explicit_combat_text:
            logger.info("Quest classified as NPC because explicit dialogue wording overrides yellow OCR")
            return QuestType.NPC

        if yellow_tokens:
            entities = parse_yellow_entities(yellow_tokens)
            if explicit_combat_text:
                logger.info("Quest classified as COMBAT because explicit combat wording overrides yellow OCR")
                return QuestType.COMBAT
            if entities.get("item"):
                logger.info(f"Quest classified as SPECIAL_NPC_SHOP via yellow entity item: '{entities['item']}'")
                return QuestType.SPECIAL_NPC_SHOP
            coherent_combat_entities = bool(
                (entities.get("monster") and int(entities.get("count") or 0) > 0)
                or (entities.get("map_name") and entities.get("monster"))
            )
            if coherent_combat_entities:
                logger.info(f"Quest classified as COMBAT via yellow entity monster='{entities.get('monster')}', map='{entities.get('map_name')}'")
                return QuestType.COMBAT
            if entities.get("npc"):
                logger.info(
                    f"Quest classified as NPC via yellow entity npc='{entities.get('npc')}', "
                    f"city='{entities.get('city')}'"
                )
                return QuestType.NPC

        if frame is None and not text:
            if hasattr(self, "_vision") and self._vision and hasattr(self._vision, "_screen") and self._vision._screen:
                frame = self._vision._screen.capture()

        # Explicit quest wording is more specific than generic sidebar icons.
        # In particular, Western Merchant quests can reuse the NPC icon.
        if text:
            norm_text = normalize_game_text(text)
            # Lọc bỏ nhiễu không phải nhiệm vụ (quá ngắn, thông báo tài khoản, cảnh báo bảo mật, thông báo số lần hôm nay)
            if len(norm_text) < 8 or any(w in norm_text for w in (
                "mat khau", "tai khoan", "cung cap thong tin", "nph se khong yeu cau",
                "doi mat khau", "thong tin vat pham", "hinh anh", "tuy chon",
                "hom nay da hoan thanh", "con co the nhan", "lan nhiem vu", "so lan nhan",
                "mai hay den", "khong the nhan them"
            )):
                return QuestType.UNKNOWN

            # Lọc bỏ các nhiệm vụ không phải Thương Hội (Tân Thủ, Thế Giới, Thiên Môn Trận, v.v.)
            if any(w in norm_text for w in ("tan thu", "huong dan tan thu", "the gioi", "thien mon tran", "nhiem vu the gioi", "nhiem vu huong dan")) and not any(w in norm_text for w in ("thuong hoi", "thuonghoi")):
                return QuestType.UNKNOWN

            # 1. Direct keyword rules on normalized text
            has_merchant = any(k in norm_text for k in ("thuong nhan tay vuc", "thuong nhan phuong tay", "thuong nhan tay ho"))
            matched_shop_item = match_western_merchant_item(text)
            has_shop_item = bool(matched_shop_item)
            has_buy_merchant = (
                any(k in norm_text for k in ("mua", "dat mua", "dat", "tim mua", "lay", "binh", "chiec"))
                and any(k in norm_text for k in ("tay vuc", "phuong tay", "tay ho", "thuong nhan"))
            )
            # Dấu hiệu chỉ đường nhiều thành phố để tìm thương nhân Tây Vực (ví dụ: 'để tìm ông ta', 'Biện Kinh, Thành Đô, Tuyền Châu...')
            has_multiple_cities = sum(
                1 for c in ("bien kinh", "thanh do", "tuyen chau", "tuong duong", "duong chau", "dai ly")
                if c in norm_text
            ) >= 2
            is_merchant_direction = (
                "de tim ong ta" in norm_text
                or (has_multiple_cities and any(k in norm_text for k in ("thuong nhan", "tay vuc", "phuong tay", "tay ho")))
            )

            if (
                has_shop_item and (has_merchant or has_buy_merchant or is_merchant_direction or any(k in norm_text for k in ("mua", "dat", "lay", "tim")))
            ) or (has_merchant and any(k in norm_text for k in ("mua", "vat pham", "dat mua"))):
                logger.info(f"Quest classified as SPECIAL_NPC_SHOP via merchant/shop keywords")
                return QuestType.SPECIAL_NPC_SHOP

            if any(kw in norm_text for kw in (
                "tieu diet", "danh bai", "ha guc", "giet", "danh quai", "di diet", "diet",
                "trung tri", "truing tri", "tao phan", "lam phan", "phan do", "thon dan tao phan", "di bat", "bat", "coc doc",
                "son tac",
            )) or (any(m in norm_text for m in ("quai", "coc doc", "nhen doc", "ca sau", "soi", "nguoi lun", "ban nuong", "son tac")) and any(c in norm_text for c in ("diet", "giet", "con", "bat", "ten"))):
                logger.info("Quest classified as COMBAT via normalized keywords")
                return QuestType.COMBAT

            # Tránh nhận nhầm nút 'Kết thúc đối thoại' thành từ khóa đối thoại NPC
            cleaned_npc_text = re.sub(r"(?:ket\s*thuc\s*doi\s*thoai|ket\s*thucdoi\s*thoai|ket\s*thuc|doi\s*thoai)", "", norm_text).strip()
            if not is_merchant_direction and any(kw in cleaned_npc_text for kw in ("gap", "noi voi", "doi thoai voi", "hoi tham", "hay den", "chuyen thu", "noi chuyen", "bao tin")):
                logger.info("Quest classified as NPC via normalized keywords")
                return QuestType.NPC

            # 2. Fuzzy matching fallback - yêu cầu ít nhất 1 từ khớp token để tránh trùng lặp ký tự ngẫu nhiên
            early_scores = []
            for quest_type, phrases in self.TEXT_RULES.items():
                best_score = 0.0
                for phrase in phrases:
                    p_tokens = set(normalize_game_text(phrase).split())
                    t_tokens = set(cleaned_npc_text.split()) if quest_type == QuestType.NPC else set(norm_text.split())
                    if p_tokens & t_tokens:
                        sc = fuzzy_score(text, phrase)
                        best_score = max(best_score, sc)
                early_scores.append((best_score, quest_type))

            if early_scores:
                early_score, early_type = max(early_scores, key=lambda item: item[0])
                if early_score >= 0.60:
                    if early_type == QuestType.NPC and is_merchant_direction:
                        pass
                    elif early_type == QuestType.SPECIAL_NPC_SHOP and not extract_shop_item(text):
                        pass  # Không phân loại là Mua đồ nếu không có tên vật phẩm cần mua
                    else:
                        logger.info(f"Quest classified as {early_type} via explicit text score={early_score:.2f}")
                        return early_type

        # Level 1: Detect icon template
        # Chỉ dùng icon template match nếu có frame VÀ có căn cứ (text hoặc yellow_tokens),
        # tránh trường hợp text rỗng nhận nhầm icon NPC của các quest khác hiển thị trên HUD.
        if frame is not None and (text or yellow_tokens):
            for template_name, quest_type in self.ICON_TEMPLATE_MAP.items():
                if not self._vision._templates.has(template_name):
                    continue
                result = self._vision.find_template(template_name, frame)
                if result.found:
                    logger.info(f"Quest classified as {quest_type} via icon '{template_name}'")
                    return quest_type

        # Level 2: UNKNOWN - phải log + không đoán bừa
        logger.warning("Quest classified as UNKNOWN – no rule or icon matched")
        return QuestType.UNKNOWN
