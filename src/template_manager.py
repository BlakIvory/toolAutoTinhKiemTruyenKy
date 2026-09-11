"""
AutoQuestTool - Template Manager
Load và cache tất cả template images từ disk.
Không đọc ảnh từ disk mỗi vòng loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from .error_codes import ErrorCode, BotError
from .config_loader import ConfigLoader, TemplateConfig


class TemplateManager:
    """
    Load template images một lần khi startup.
    Cung cấp interface để lấy template đã load.
    """

    def __init__(self, config_loader: ConfigLoader, base_dir: Path = Path(".")):
        self._loader = config_loader
        self._base_dir = base_dir
        # Cache: name → (image_bgr, config)
        self._cache: Dict[str, Tuple[np.ndarray, TemplateConfig]] = {}
        self._loaded = False

    def load_all(self) -> None:
        """Load tất cả template đã khai báo trong templates.yaml."""
        templates_cfg = self._loader.load_templates()
        missing = []
        for name, tcfg in templates_cfg.templates.items():
            path = self._base_dir / tcfg.path
            if not path.exists():
                logger.warning(f"Template image not found: {path} ('{name}') – skipping")
                missing.append(name)
                continue
            img = cv2.imread(str(path))
            if img is None:
                logger.warning(f"Failed to load template image: {path}")
                missing.append(name)
                continue
            self._cache[name] = (img, tcfg)
            logger.debug(f"Template loaded: {name} ({img.shape[1]}×{img.shape[0]})")

        loaded_count = len(self._cache)
        total_count = len(templates_cfg.templates)
        logger.info(
            f"TemplateManager: {loaded_count}/{total_count} templates loaded"
            + (f" (missing: {missing})" if missing else "")
        )
        self._loaded = True

    def get(self, name: str) -> Tuple[np.ndarray, TemplateConfig]:
        """Trả về (image, config) của template. Raise nếu không tìm thấy."""
        if not self._loaded:
            self.load_all()
        if name not in self._cache:
            raise BotError(
                ErrorCode.VIS_002,
                f"Template '{name}' not loaded. Check templates.yaml and asset path.",
            )
        return self._cache[name]

    def get_image(self, name: str) -> np.ndarray:
        img, _ = self.get(name)
        return img

    def get_config(self, name: str) -> TemplateConfig:
        _, cfg = self.get(name)
        return cfg

    def has(self, name: str) -> bool:
        return name in self._cache

    def load_single(self, name: str, path: str, confidence: float = 0.85) -> None:
        """Load thêm template không có trong config (dùng cho test)."""
        img = cv2.imread(path)
        if img is None:
            raise BotError(ErrorCode.VIS_002, f"Cannot load image: {path}")
        from .config_loader import TemplateConfig
        self._cache[name] = (img, TemplateConfig(path=path, confidence=confidence))

    @property
    def loaded_names(self):
        return list(self._cache.keys())
