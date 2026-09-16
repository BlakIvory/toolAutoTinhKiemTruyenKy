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
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def fuzzy_score(observed: str, expected: str) -> float:
    """Return 0..1; tolerate missing accents and imperfect OCR tokens."""
    left, right = normalize_game_text(observed), normalize_game_text(expected)
    if not left or not right:
        return 0.0
    sequence = SequenceMatcher(None, left, right).ratio()
    left_tokens, right_tokens = set(left.split()), set(right.split())
    token = len(left_tokens & right_tokens) / max(1, len(right_tokens))
    contains = 1.0 if left in right or right in left else 0.0
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
    """Reject unrelated place names that happen to be fuzzy string matches."""
    value = normalize_game_text(text)
    tokens = set(value.split())
    has_merchant = "thuong" in tokens and "nhan" in tokens
    has_west = "tay" in tokens
    has_region = "vuc" in tokens or "ho" in tokens or "phuong" in tokens
    return has_merchant and has_west and has_region


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
            x1, y1, x2, y2 = region
            image = frame[y1:y2, x1:x2]
        engine = self._get_engine()
        if engine is None or image.size == 0:
            return []
        try:
            result, _ = engine(image)
        except Exception as exc:
            logger.warning(f"OCR read failed: {exc}")
            return []
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
