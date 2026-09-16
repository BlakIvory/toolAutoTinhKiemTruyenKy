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

KNOWN_DIALOGUE_NPCS = (
    "Bách Hiểu Sinh",
    "Vệ binh thành môn",
    "Vệ Binh Thành Môn",
    "Vân Du đạo nhân",
    "Nạn dân Đinh",
    "Khẩu Chuẩn",
    "Phạm Trọng Yêm",
    "Vương Nghiệp Vi",
    "Hành Cước Thương Nhân",
    "Dã Tẩu",
    "La Sa",
    "Lâm Dược Sư",
    "Thái thú",
    "Bộ đầu",
    "Tiểu đồng",
    "Lão đài",
    "Hạ Kiệt",
    "Uy Nhĩ Tốn",
    "Chủ Bàn Cử",
    "Tiêu sứ",
    "Sứ giả môn phái",
    "Quản lý cửa hiệu",
    "Đại Chí Cường",
    "Dụ Chân Nhân",
    "Thủ kho Bắc",
    "Thủ kho Nam",
    "Thủ kho Đông",
    "Thủ kho Tây",
    "Xa phu Bắc",
    "Xa phu Nam",
    "Chủ Tạp hóa",
    "Chủ Tiệm Nam phục",
    "Chủ Tiệm Nữ phục",
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
    "Sứ giả Tàng Kiếm",
    "Sứ giả bang phái",
    "Lễ quan",
    "Lý Phong Thu",
)


def get_canonical_npc_name(npc: str, original_text: str = "") -> str:
    """Restore proper Vietnamese accents and capitalization for known NPC names."""
    if not npc:
        return ""
    norm_npc = normalize_game_text(npc)
    for known in KNOWN_DIALOGUE_NPCS:
        if normalize_game_text(known) == norm_npc:
            return known

    # Fuzzy match with strict threshold
    scored = [(fuzzy_score(norm_npc, normalize_game_text(k)), k) for k in KNOWN_DIALOGUE_NPCS]
    if scored:
        sc, best_k = max(scored, key=lambda x: x[0])
        if sc >= 0.82:
            return best_k

    if original_text:
        import re
        orig_match = re.search(
            r"(?:gặp|tìm|nói với|đối thoại với)\s+([^:,.!?\n\r]+?)(?=\s+nói|\s+tại|\s+để|\s+rồi|\s+và|\s+báo|\s+hàng|:|,|\.|$)",
            original_text, re.IGNORECASE
        )
        if orig_match and orig_match.group(1).strip():
            return orig_match.group(1).strip()

    return npc.title()


def extract_dialogue_target(text: str, with_accents: bool = False) -> tuple[str, str]:
    """
    Extract approximate city/NPC names from a commerce quest description.
    If with_accents=True, restores proper Vietnamese diacritics and capitalization for NPC.
    """
    import re

    normalized = normalize_game_text(text)
    city = ""
    # Mẫu tìm thành phố / bản đồ cần đến
    city_match = re.search(
        r"(?:hay den|den|toi)\s+([a-z0-9 ]+?)(?=\s+noi voi|\s+gap|\s+tim|\s+de|\s+hoi|,|\.|$)",
        normalized
    )
    if city_match:
        city = city_match.group(1).strip()

    if not city:
        for pattern in (
            r"(?:den|toi) ([a-z0-9 ]+?)(?= noi voi | gap | tim | de |$)",
            r"tai ([a-z0-9 ]+?)(?= noi voi | gap | tim | co |$)",
        ):
            match = re.search(pattern, normalized)
            if match:
                city = match.group(1).strip()
                break

    # Canonicalize known cities / special place names (e.g. Thành Đô Phủ -> Thành Đô)
    norm_city = normalize_game_text(city)
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

    npc = ""
    # Mẫu tìm tên NPC: gap [Tên NPC] noi ...
    npc_match = re.search(
        r"(?:gap|tim|noi voi|doi thoai voi)\s+([a-z0-9 ]+?)(?=\s+noi|\s+tai|\s+de|\s+roi|\s+va|\s+bao|\s+hang|:|,|\.|$)",
        normalized
    )
    if npc_match:
        npc = npc_match.group(1).strip()
    else:
        for pattern in (
            r"(?:noi|doi thoai) voi ([a-z0-9 ]+?)(?= hang | de | roi | va | tai |:|$)",
            r"(?:gap|tim) ([a-z0-9 ]+?)(?= tai | de | roi | va |:|$)",
        ):
            match = re.search(pattern, normalized)
            if match:
                npc = match.group(1).strip()
                break

    if with_accents and npc:
        npc = get_canonical_npc_name(npc, original_text=text)

    return city, npc


def extract_shop_item(text: str) -> str:
    """Extract the requested item from Western Merchant quest wording."""
    import re

    value = normalize_game_text(text)
    patterns = (
        r"dat (?:mot |[0-9]+ )?(?:binh |chiec |cai |vien |cuon |tam |mon |hop |bao |goi )?(.+?)(?:\s+o cho thuong nhan|\s+tai thuong nhan|\s+tu thuong nhan|\s+o cho|\s+tai|\s+tu|\s+xin giup)",
        r"(?:mua|lay|tim mua)\s+(?:mot |[0-9]+\s+)?(?:binh |chiec |cai |vien |cuon |tam |mon |hop |bao |goi )?([a-z0-9 ]+?)(?:\s+o cho|\s+tu|\s+tai|\s+ve|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            item = match.group(1).strip()
            for prefix in ("mot ", "binh ", "chiec ", "cai ", "vien ", "cuon ", "tam ", "mon "):
                if item.startswith(prefix):
                    item = item[len(prefix):].strip()
            if item:
                return item
    return ""


def extract_combat_target(text: str) -> tuple[str, str, int]:
    """Extract map, monster name, and required count from combat quest wording."""
    import re

    normalized = normalize_game_text(text)
    # Ví dụ: "Hãy đến Tường Vân động tiêu diệt 20 Sói điên"
    match = re.search(
        r"(?:hay den|den|toi)\s+([a-z0-9 ]+?)\s+(?:tieu diet|danh bai|ha guc|giet)\s+(\d+)\s+([a-z0-9 ]+?)(?:\.|$|,|\s+dem|\s+ve)",
        normalized
    )
    if match:
        map_name = match.group(1).strip()
        count = int(match.group(2).strip())
        monster = match.group(3).strip()
        return map_name, monster, count

    # Fallback pattern
    count_match = re.search(r"(\d+)", normalized)
    count = int(count_match.group(1)) if count_match else 20
    return "", "", count


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
            "mua vật phẩm", "mua đồ", "tìm mua",
        ),
        QuestType.COMBAT: (
            "đánh quái", "tiêu diệt", "đánh bại", "giết", "số lượng",
        ),
        QuestType.NPC: (
            "đối thoại", "nói chuyện", "gặp", "tìm npc", "hỏi thăm",
        ),
    }

    def __init__(self, vision: VisionEngine, config: AppConfig):
        self._vision = vision
        self.config = config

    def classify(self, frame: Optional[np.ndarray] = None, text: str = "") -> QuestType:
        """
        Phân loại quest type từ frame hiện tại hoặc quest text.
        Priority: text classification (với validation nghiêm ngặt) → icon template → UNKNOWN.
        """
        if frame is None:
            frame = self._vision._screen.capture()

        # Explicit quest wording is more specific than generic sidebar icons.
        # In particular, Western Merchant quests can reuse the NPC icon.
        if text:
            norm_text = normalize_game_text(text)
            # Lọc bỏ nhiễu không phải nhiệm vụ (quá ngắn, thông báo tài khoản, cảnh báo bảo mật)
            if len(norm_text) < 8 or any(w in norm_text for w in (
                "mat khau", "tai khoan", "cung cap thong tin", "nph se khong yeu cau",
                "doi mat khau", "thong tin vat pham", "hinh anh", "tuy chon"
            )):
                return QuestType.UNKNOWN

            # 1. Direct keyword rules on normalized text
            has_merchant = any(k in norm_text for k in ("thuong nhan tay vuc", "thuong nhan phuong tay", "thuong nhan tay ho"))
            has_buy_merchant = any(k in norm_text for k in ("mua", "dat mua", "tim mua", "lay")) and any(k in norm_text for k in ("tay vuc", "phuong tay", "tay ho", "thuong nhan"))
            if has_merchant or has_buy_merchant:
                logger.info("Quest classified as SPECIAL_NPC_SHOP via Western Merchant keywords")
                return QuestType.SPECIAL_NPC_SHOP

            if any(kw in norm_text for kw in ("tieu diet", "danh bai", "ha guc", "giet", "danh quai")):
                logger.info("Quest classified as COMBAT via normalized keywords")
                return QuestType.COMBAT

            if any(kw in norm_text for kw in ("gap", "noi voi", "doi thoai", "hoi tham", "hay den", "chuyen thu", "noi chuyen")):
                logger.info("Quest classified as NPC via normalized keywords")
                return QuestType.NPC

            # 2. Fuzzy matching fallback - yêu cầu ít nhất 1 từ khớp token để tránh trùng lặp ký tự ngẫu nhiên
            early_scores = []
            for quest_type, phrases in self.TEXT_RULES.items():
                best_score = 0.0
                for phrase in phrases:
                    p_tokens = set(normalize_game_text(phrase).split())
                    t_tokens = set(norm_text.split())
                    if p_tokens & t_tokens:
                        sc = fuzzy_score(text, phrase)
                        best_score = max(best_score, sc)
                early_scores.append((best_score, quest_type))

            if early_scores:
                early_score, early_type = max(early_scores, key=lambda item: item[0])
                if early_score >= 0.60:
                    if early_type == QuestType.SPECIAL_NPC_SHOP and not extract_shop_item(text):
                        pass  # Không phân loại là Mua đồ nếu không có tên vật phẩm cần mua
                    else:
                        logger.info(f"Quest classified as {early_type} via explicit text score={early_score:.2f}")
                        return early_type

        # Level 1: Detect icon template
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
