"""
AutoQuestTool - Vision Engine
Computer Vision dựa trên OpenCV template matching.
Priority: tọa độ cố định → template matching → OCR → AI
"""

from __future__ import annotations

import time
from typing import Optional, List, Tuple

import cv2
import numpy as np
from loguru import logger

from .error_codes import ErrorCode, BotError
from .config_loader import AppConfig, ConfigLoader
from .template_manager import TemplateManager
from .screen_capture import ScreenCapture


# Kết quả template match
class MatchResult:
    def __init__(
        self,
        found: bool,
        x: int = 0,
        y: int = 0,
        w: int = 0,
        h: int = 0,
        confidence: float = 0.0,
        center_x: int = 0,
        center_y: int = 0,
    ):
        self.found = found
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.confidence = confidence
        self.center_x = center_x
        self.center_y = center_y

    def __bool__(self):
        return self.found

    def __repr__(self):
        if self.found:
            return (
                f"MatchResult(found=True, center=({self.center_x},{self.center_y}), "
                f"confidence={self.confidence:.3f})"
            )
        return "MatchResult(found=False)"


class VisionEngine:
    """
    OpenCV-based template matching engine.
    
    API:
        find_template(name, frame, region)
        find_all_templates(name, frame, region)
        template_exists(name, frame, region) → bool
        wait_for_template(name, timeout) → MatchResult
        screen_matches(name, frame) → bool
        get_match_confidence(name, frame) → float
    """

    MATCH_METHODS = {
        "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
        "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
        "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
    }

    def __init__(
        self,
        config: AppConfig,
        config_loader: ConfigLoader,
        template_manager: TemplateManager,
        screen_capture: ScreenCapture,
    ):
        self.config = config
        self._loader = config_loader
        self._templates = template_manager
        self._screen = screen_capture
        method_str = config.vision.match_method
        self._method = self.MATCH_METHODS.get(method_str, cv2.TM_CCOEFF_NORMED)
        self._default_confidence = config.vision.default_confidence

    # ============================
    # Core matching
    # ============================

    def _get_region_coords(
        self, region_name: Optional[str]
    ) -> Optional[Tuple[int, int, int, int]]:
        if not region_name:
            return None
        try:
            return self._loader.get_region(region_name)
        except BotError:
            logger.warning(f"Region '{region_name}' not found in ui_coordinates.yaml")
            return None

    def _match(
        self,
        frame: np.ndarray,
        template_img: np.ndarray,
        confidence_threshold: float,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> MatchResult:
        """Core template matching. Trả về best match."""
        if region:
            x1, y1, x2, y2 = region
            search_area = frame[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            search_area = frame
            offset_x, offset_y = 0, 0

        # Grayscale matching cho nhanh hơn
        gray_frame = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
        gray_tmpl = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

        if gray_frame.shape[0] < gray_tmpl.shape[0] or gray_frame.shape[1] < gray_tmpl.shape[1]:
            return MatchResult(found=False)

        result = cv2.matchTemplate(gray_frame, gray_tmpl, self._method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if self._method == cv2.TM_SQDIFF_NORMED:
            confidence = 1.0 - min_val
            loc = min_loc
        else:
            confidence = max_val
            loc = max_loc

        if confidence >= confidence_threshold:
            tx, ty = loc
            tw, th = gray_tmpl.shape[1], gray_tmpl.shape[0]
            abs_x = offset_x + tx
            abs_y = offset_y + ty
            return MatchResult(
                found=True,
                x=abs_x, y=abs_y,
                w=tw, h=th,
                confidence=confidence,
                center_x=abs_x + tw // 2,
                center_y=abs_y + th // 2,
            )
        return MatchResult(found=False, confidence=confidence)

    # ============================
    # Public API
    # ============================

    def find_template(
        self,
        name: str,
        frame: Optional[np.ndarray] = None,
        region_override: Optional[Tuple[int, int, int, int]] = None,
    ) -> MatchResult:
        """
        Tìm template trong frame.
        Nếu frame=None → chụp mới.
        region_override = (x1, y1, x2, y2) để override region trong config.
        """
        if frame is None:
            frame = self._screen.capture()

        tmpl_img, tmpl_cfg = self._templates.get(name)
        confidence = tmpl_cfg.confidence or self._default_confidence

        # Ưu tiên region_override, sau đó region từ config
        if region_override:
            region = region_override
        elif tmpl_cfg.region:
            region = self._get_region_coords(tmpl_cfg.region)
        else:
            region = None

        result = self._match(frame, tmpl_img, confidence, region)
        if result.found:
            logger.debug(f"find_template '{name}': FOUND at ({result.center_x},{result.center_y}) conf={result.confidence:.3f}")
        else:
            logger.debug(f"find_template '{name}': NOT found (best conf={result.confidence:.3f})")
        return result

    def find_all_templates(
        self,
        name: str,
        frame: Optional[np.ndarray] = None,
        region_override: Optional[Tuple[int, int, int, int]] = None,
        max_count: int = 20,
    ) -> List[MatchResult]:
        """Tìm tất cả occurences của template trong frame."""
        if frame is None:
            frame = self._screen.capture()

        tmpl_img, tmpl_cfg = self._templates.get(name)
        confidence = tmpl_cfg.confidence or self._default_confidence

        if region_override:
            region = region_override
        elif tmpl_cfg.region:
            region = self._get_region_coords(tmpl_cfg.region)
        else:
            region = None

        if region:
            x1, y1, x2, y2 = region
            search_area = frame[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            search_area = frame
            offset_x, offset_y = 0, 0

        gray_frame = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
        gray_tmpl = cv2.cvtColor(tmpl_img, cv2.COLOR_BGR2GRAY)

        result_map = cv2.matchTemplate(gray_frame, gray_tmpl, self._method)
        tw, th = gray_tmpl.shape[1], gray_tmpl.shape[0]

        results = []
        mask = result_map.copy()

        for _ in range(max_count):
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(mask)
            if self._method == cv2.TM_SQDIFF_NORMED:
                conf = 1.0 - min_val
                loc = min_loc
            else:
                conf = max_val
                loc = max_loc

            if conf < confidence:
                break

            tx, ty = loc
            abs_x = offset_x + tx
            abs_y = offset_y + ty
            results.append(MatchResult(
                found=True,
                x=abs_x, y=abs_y,
                w=tw, h=th,
                confidence=conf,
                center_x=abs_x + tw // 2,
                center_y=abs_y + th // 2,
            ))
            # Mask out vùng đã tìm thấy
            cv2.rectangle(
                mask,
                (max(0, tx - tw // 2), max(0, ty - th // 2)),
                (min(mask.shape[1], tx + tw), min(mask.shape[0], ty + th)),
                0, -1
            )

        logger.debug(f"find_all_templates '{name}': {len(results)} found")
        return results

    def template_exists(
        self,
        name: str,
        frame: Optional[np.ndarray] = None,
    ) -> bool:
        return bool(self.find_template(name, frame))

    def wait_for_template(
        self,
        name: str,
        timeout: Optional[float] = None,
        interval: Optional[float] = None,
        take_screenshot_on_fail: bool = True,
    ) -> MatchResult:
        """
        Chờ template xuất hiện trong timeout giây.
        Raise BotError nếu hết timeout.
        """
        t = timeout or self.config.timing.action_timeout
        iv = interval or self.config.timing.screenshot_interval
        deadline = time.time() + t
        logger.debug(f"wait_for_template '{name}' timeout={t}s")

        while time.time() < deadline:
            frame = self._screen.capture()
            result = self.find_template(name, frame)
            if result:
                return result
            time.sleep(iv)

        # Timeout
        if take_screenshot_on_fail:
            try:
                frame = self._screen.capture()
                self._screen.save_debug_image(frame, f"timeout_{name}")
            except Exception:
                pass

        raise BotError(
            ErrorCode.VIS_001,
            f"Template '{name}' not found within {t}s",
        )

    def wait_for_any_template(
        self,
        names: List[str],
        timeout: Optional[float] = None,
        interval: Optional[float] = None,
    ) -> Tuple[str, MatchResult]:
        """
        Chờ BẤT KỲ template nào trong danh sách xuất hiện.
        Trả về (name, MatchResult).
        """
        t = timeout or self.config.timing.action_timeout
        iv = interval or self.config.timing.screenshot_interval
        deadline = time.time() + t

        while time.time() < deadline:
            frame = self._screen.capture()
            for name in names:
                result = self.find_template(name, frame)
                if result:
                    logger.debug(f"wait_for_any_template: '{name}' found")
                    return (name, result)
            time.sleep(iv)

        raise BotError(
            ErrorCode.VIS_001,
            f"None of {names} found within {t}s",
        )

    def screen_matches(
        self,
        name: str,
        frame: Optional[np.ndarray] = None,
    ) -> bool:
        """Alias cho template_exists."""
        return self.template_exists(name, frame)

    def get_match_confidence(
        self,
        name: str,
        frame: Optional[np.ndarray] = None,
    ) -> float:
        """Trả về confidence của best match (dù không đủ threshold)."""
        if frame is None:
            frame = self._screen.capture()
        tmpl_img, tmpl_cfg = self._templates.get(name)
        # Không dùng threshold → luôn trả về confidence
        result = self._match(frame, tmpl_img, -1.0)
        return result.confidence

    def detect_current_screen(self, screen_names: List[str]) -> Optional[str]:
        """
        Xác định màn hình hiện tại trong danh sách screen_names.
        Trả về tên màn hình đầu tiên match, hoặc None.
        """
        frame = self._screen.capture()
        for name in screen_names:
            if self.template_exists(name, frame):
                logger.info(f"Current screen detected: {name}")
                return name
        logger.warning(f"Current screen unknown (checked: {screen_names})")
        return None
