"""OCR and accent-insensitive fuzzy matching for game UI text."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Optional

import numpy as np
from loguru import logger


def normalize_game_text(value: str) -> str:
    value = unicodedata.normalize("NFD", value or "")
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.casefold().replace("đ", "d")
    clean = " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())
    # Sửa một số lỗi OCR điển hình do font game Kiếm Hiệp
    clean = re.sub(r"\bdugng\b", "duong", clean)
    clean = re.sub(r"\bdudng\b", "duong", clean)
    clean = re.sub(r"\btugng\b", "tuong", clean)
    clean = re.sub(r"\btupen\b", "tuyen", clean)
    clean = re.sub(r"\bngi\b", "ngu", clean)
    clean = re.sub(r"\bngui\b", "ngu", clean)
    clean = re.sub(r"\bn[6o]ivoi\b", "noi voi", clean)
    clean = re.sub(r"\bluic\b", "luc", clean)
    clean = re.sub(r"\blyc\b", "luc", clean)
    clean = re.sub(r"\bsuic\b", "suc", clean)
    clean = re.sub(r"\bsuicluc\b", "suc luc", clean)
    clean = re.sub(r"\bsuclyc\b", "suc luc", clean)
    clean = re.sub(r"\bsucluc\b", "suc luc", clean)
    clean = re.sub(r"\btaodai\b", "tao dai", clean)
    clean = re.sub(r"\bthanh\s*d[80]\b", "thanh do", clean)
    clean = re.sub(r"\bthanh\s*d[80]\s*phi\b", "thanh do phu", clean)
    clean = re.sub(r"\bthanh\s*do\s*phi\b", "thanh do phu", clean)
    return clean


def fuzzy_score(observed: str, expected: str) -> float:
    """Return 0..1; tolerate missing accents and imperfect OCR tokens."""
    left, right = normalize_game_text(observed), normalize_game_text(expected)
    if not left or not right:
        return 0.0
    sequence = SequenceMatcher(None, left, right).ratio()
    left_tokens, right_tokens = set(left.split()), set(right.split())
    token = len(left_tokens & right_tokens) / max(1, len(right_tokens))
    min_len = min(len(left), len(right))
    max_len = max(len(left), len(right))
    exact_phrase = (f" {left} " in f" {right} " or f" {right} " in f" {left} ")
    if exact_phrase:
        contains = 1.0
    elif (left in right or right in left) and (min_len >= 0.70 * max_len):
        contains = 1.0
    else:
        contains = 0.0
    # Token coverage carries more weight for long quest descriptions: OCR often
    # surrounds the useful phrase with unrelated UI text.
    return max(sequence, 0.30 * sequence + 0.70 * token, contains)


def is_explicit_quest_exhausted(text: str) -> bool:
    """Require completion wording; a generic 'maximum 20/day' notice is not enough."""
    value = normalize_game_text(text)
    return any((
        "khong the nhan them" in value,
        "da hoan thanh 20" in value,
        "da nhan toi da" in value,
        "het nhiem vu thuong hoi" in value,
    ))


def is_western_merchant_name(text: str) -> bool:
    """Reject unrelated place names and dialogue sentences that happen to be fuzzy string matches."""
    value = normalize_game_text(text)
    words = value.split()
    if len(words) > 4 or len(value) > 25:
        return False
    if any(w in words for w in ("xin", "giup", "lay", "ve", "den", "dat", "cho", "mua", "bao", "dao", "co", "the")):
        return False
    compact = value.replace(" ", "")
    has_merchant = "thuongnhan" in compact or ("thuong" in value and "nhan" in value)
    has_west = "tay" in compact
    has_region = "vuc" in compact or "ho" in compact or "phuong" in compact
    return (has_merchant and has_west) or (has_merchant and has_region) or ("tayvuc" in compact and "thuong" in compact)


KNOWN_CITIES = (
    "Tuyền Châu", "Biện Kinh", "Thành Đô", "Tương Dương",
    "Dương Châu", "Đại Lý", "Lâm An", "Phượng Tường"
)

AMBIGUOUS_CITY_WORDS = {"duong", "chau", "tuong", "thanh", "do", "kinh", "ly", "an", "trung", "dong", "nam", "bac", "tay"}


EXACT_CITY_PATTERNS = {
    "thanh do": "Thành Đô",
    "bien kinh": "Biện Kinh",
    "tuong duong": "Tương Dương",
    "duong chau": "Dương Châu",
    "tuyen chau": "Tuyền Châu",
    "dai ly": "Đại Lý",
    "phuong tuong": "Phượng Tường",
    "lam an": "Lâm An",
}

KNOWN_TOWNS = {
    "ba lang huyen": "Ba Lăng huyện",
    "ba lang": "Ba Lăng huyện",
    "giang tan thon": "Giang Tân thôn",
    "giang tan": "Giang Tân thôn",
    "vinh lac tran": "Vĩnh Lạc trấn",
    "vinh lac": "Vĩnh Lạc trấn",
    "chu tien tran": "Chu Tiên trấn",
    "chu tien": "Chu Tiên trấn",
    "dao huong thon": "Đạo Hương thôn",
    "dao huong": "Đạo Hương thôn",
    "long tuyen thon": "Long Tuyền thôn",
    "long tuyen": "Long Tuyền thôn",
    "long mon tran": "Long Môn trấn",
    "long mon": "Long Môn trấn",
    "thach co tran": "Thạch Cổ trấn",
    "thach co": "Thạch Cổ trấn",
    "hanh hoa thon": "Hạnh Hoa thôn",
    "hanh hoa": "Hạnh Hoa thôn",
}


def match_known_town(text: Optional[str]) -> Optional[str]:
    """Nhận diện các Thôn / Trấn trong game Võ Lâm."""
    if not text:
        return None
    norm = normalize_game_text(text)
    if not norm:
        return None
    for k, v in KNOWN_TOWNS.items():
        if k == norm or norm.startswith(k + " ") or norm.endswith(" " + k):
            return v
    return None


def match_known_city(text: Optional[str]) -> Optional[str]:
    """
    Nhận diện tên thành phố chuẩn từ text OCR:
    - Bắt buộc KHỚP CHÍNH XÁC từng từ với tên thành phố hợp lệ (Thành Đô, Biện Kinh, Tương Dương...).
    - Tuyệt đối loại bỏ các map ngoại thành có hướng (Nam Thành Đô, Bắc Tuyền Châu, Đông Biện Kinh...).
    - Tuyệt đối không match từ đơn gây nhập nhằng như 'Duong', 'Chau', 'Tuong'.
    - Cho phép dung sai ký tự nhẹ (ví dụ OCR lỗi 1 chữ 'TupenChau' thay vì 'TuyenChau') khi đúng số lượng từ.
    """
    if not text:
        return None
    norm = normalize_game_text(text)
    if not norm:
        return None

    # Nếu text là từ đơn mơ hồ -> Không thể xác định chắc chắn là thành nào
    norm_compact = norm.replace(" ", "")
    if norm in AMBIGUOUS_CITY_WORDS or norm_compact in AMBIGUOUS_CITY_WORDS:
        return None

    words = norm.split()
    directions = {"nam", "bac", "dong", "tay"}
    gates = {"mon", "cong", "cua"}
    has_direction = bool(set(words) & directions)
    has_gate = bool(set(words) & gates)

    # Map dã ngoại/ngoại thành (Nam Thành Đô, Bắc Tuyền Châu...) -> Bị loại bỏ 100%
    if has_direction and not has_gate:
        return None

    # 1. Khớp chính xác 100% từng từ (Exact Word Match)
    if norm in EXACT_CITY_PATTERNS:
        return EXACT_CITY_PATTERNS[norm]

    # Khớp chính xác dạng viết liền (ví dụ OCR dính chữ: 'thanhdo', 'tuyenchau')
    compact_patterns = {k.replace(" ", ""): v for k, v in EXACT_CITY_PATTERNS.items()}
    if norm_compact in compact_patterns:
        return compact_patterns[norm_compact]

    # Khớp cụm từ chính xác (ví dụ text chứa cả tên thành + tọa độ/tên NPC: 'Tuong Duong P189/190...')
    for pattern_norm, canonical in EXACT_CITY_PATTERNS.items():
        c_words = pattern_norm.split()
        k = len(c_words)
        if len(words) >= k:
            for i in range(len(words) - k + 1):
                if i > 0 and words[i - 1] in directions:
                    continue
                cand = " ".join(words[i:i + k])
                if cand == pattern_norm:
                    return canonical

    # 2. Xử lý trường hợp có từ phụ cổng thành hợp lệ (ví dụ: 'Bac mon Dudng Chau')
    if has_gate:
        for pattern_norm, canonical in EXACT_CITY_PATTERNS.items():
            c_words = pattern_norm.split()
            k = len(c_words)
            if len(words) >= k:
                for i in range(len(words) - k + 1):
                    cand = " ".join(words[i:i + k])
                    if cand == pattern_norm or SequenceMatcher(None, cand, pattern_norm).ratio() >= 0.80:
                        return canonical

    # 3. Dung sai OCR nhẹ (chỉ cho phép khi độ dài tương đương và không bị thừa từ)
    # Ví dụ: 'TupenChau' (sai 1 chữ so với 'tuyenchau', ratio >= 0.85)
    if len(words) <= 2:
        for pattern_norm, canonical in EXACT_CITY_PATTERNS.items():
            pat_compact = pattern_norm.replace(" ", "")
            if abs(len(norm_compact) - len(pat_compact)) <= 1:
                ratio = SequenceMatcher(None, norm_compact, pat_compact).ratio()
                if ratio >= 0.85:
                    return canonical

    return None


def is_same_city(city_a: Optional[str], city_b: Optional[str]) -> bool:
    """
    So sánh xem 2 tên thành phố/map có phải là cùng một thành phố hay không.
    - Ngăn chặn tuyệt đối việc nhận nhầm 'Dương Châu' và 'Tương Dương' khi chỉ có từ 'Duong'.
    - Ngăn chặn tuyệt đối việc nhận nhầm map phụ có hướng (Nam Thành Đô) với thành phố chính (Thành Đô).
    """
    if not city_a or not city_b:
        return False

    norm_a = normalize_game_text(city_a)
    norm_b = normalize_game_text(city_b)

    if not norm_a or not norm_b:
        return False

    # Nếu một bên có hướng độc lập (không đi kèm môn/cổng) mà bên kia không có -> Khác thành phố hoàn toàn
    directions = {"nam", "bac", "dong", "tay"}
    gates = {"mon", "cong", "cua"}
    words_a = set(norm_a.split())
    words_b = set(norm_b.split())
    dir_a = (words_a & directions) if not (words_a & gates) else set()
    dir_b = (words_b & directions) if not (words_b & gates) else set()
    if dir_a != dir_b:
        return False

    canon_a = match_known_city(city_a)
    canon_b = match_known_city(city_b)

    # Nếu cả 2 đều chuẩn hóa được về thành phố đã biết
    if canon_a and canon_b:
        return canon_a == canon_b

    # Nếu một bên là từ đơn mơ hồ (vd: "duong", "chau", "tuong") -> TUYỆT ĐỐI KHÔNG MATCH
    if norm_a in AMBIGUOUS_CITY_WORDS or norm_b in AMBIGUOUS_CITY_WORDS:
        return False

    if norm_a == norm_b or norm_a.replace(" ", "") == norm_b.replace(" ", ""):
        return True

    # Đối với các map khác ngoài KNOWN_CITIES (map quái, map phụ)
    if len(norm_a) >= 6 and len(norm_b) >= 6:
        if norm_a == norm_b:
            return True
        if SequenceMatcher(None, norm_a, norm_b).ratio() >= 0.75:
            return True

    return False


def is_safe_map_result(observed: str, requested: str) -> bool:
    """Do not treat Nam/Bac/Tay/Dong submaps as their parent city."""
    seen = set(normalize_game_text(observed).split())
    wanted = set(normalize_game_text(requested).split())
    directions = {"nam", "bac", "tay", "dong"}
    return bool(seen and wanted) and not ((seen & directions) - (wanted & directions))


@dataclass(frozen=True)
class TextBox:
    text: str
    confidence: float
    center_x: int
    center_y: int


class GameTextRecognizer:
    """Lazy RapidOCR wrapper. The bot remains usable when OCR is unavailable."""

    def __init__(self) -> None:
        self._engine = None
        self._load_attempted = False

    def _get_engine(self):
        if not self._load_attempted:
            self._load_attempted = True
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._engine = RapidOCR()
            except Exception as exc:
                logger.warning(f"OCR unavailable: {exc}")
        return self._engine

    def read(self, frame: np.ndarray, region: Optional[tuple[int, int, int, int]] = None) -> list[TextBox]:
        if frame is None:
            return []
        x1 = y1 = 0
        image = frame
        if region:
            rx1, ry1, rx2, ry2 = region
            h, w = frame.shape[:2]
            # UI coordinates in this project use a 1600x900 reference frame.
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
            image = frame[y1:y2, x1:x2]
        engine = self._get_engine()
        if engine is None or image.size == 0:
            return []
        try:
            result, _ = engine(image)
        except Exception as exc:
            logger.warning(f"OCR read failed: {exc}")
            import gc
            gc.collect()
            time.sleep(0.3)
            try:
                result, _ = engine(image)
            except Exception:
                return []

        if not hasattr(self, "_call_count"):
            self._call_count = 0
        self._call_count += 1
        if self._call_count % 10 == 0:
            import gc
            gc.collect()

        boxes: list[TextBox] = []
        for item in result or []:
            polygon, text, confidence = item
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            boxes.append(TextBox(str(text), float(confidence), int(sum(xs) / len(xs)) + x1, int(sum(ys) / len(ys)) + y1))
        return boxes

    @staticmethod
    def best_match(boxes: Iterable[TextBox], phrases: Iterable[str], minimum: float = 0.58):
        best = None
        for box in boxes:
            for phrase in phrases:
                score = fuzzy_score(box.text, phrase) * max(0.5, box.confidence)
                if best is None or score > best[0]:
                    best = (score, box, phrase)
        return best if best and best[0] >= minimum else None
