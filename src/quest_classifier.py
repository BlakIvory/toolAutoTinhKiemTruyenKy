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

    def __init__(self, vision: VisionEngine, config: AppConfig):
        self._vision = vision
        self.config = config

    def classify(self, frame: Optional[np.ndarray] = None) -> QuestType:
        """
        Phân loại quest type từ frame hiện tại.
        Priority: icon template → UNKNOWN.
        """
        if frame is None:
            frame = self._vision._screen.capture()

        # Level 1: Detect icon template
        for template_name, quest_type in self.ICON_TEMPLATE_MAP.items():
            if not self._vision._templates.has(template_name):
                continue
            result = self._vision.find_template(template_name, frame)
            if result.found:
                logger.info(f"Quest classified as {quest_type} via icon '{template_name}'")
                return quest_type

        # Level 2: UNKNOWN - phải log + skip nguy hiểm
        logger.warning("Quest classified as UNKNOWN – no icon matched")
        return QuestType.UNKNOWN
