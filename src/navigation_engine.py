"""
AutoQuestTool - Navigation Engine
Điều hướng nhân vật trên bản đồ.
Ưu tiên deterministic navigation (không dùng ML).
"""

from __future__ import annotations

import time
from typing import Optional, List, Tuple

from loguru import logger

from .config_loader import ConfigLoader, AppConfig
from .input_controller import InputController
from .vision_engine import VisionEngine
from .error_codes import ErrorCode, BotError


class NavigationEngine:
    """
    Quản lý navigation trong game.
    API:
        go_to_map(map_id)
        go_to_coordinate(x, y)
        go_to_npc(npc_id)
        go_to_shop(shop_id)
        return_to_quest_giver()
        get_current_map() → str
    """

    def __init__(
        self,
        config: AppConfig,
        config_loader: ConfigLoader,
        input_ctrl: InputController,
        vision: VisionEngine,
    ):
        self.config = config
        self._loader = config_loader
        self._input = input_ctrl
        self._vision = vision
        self._current_map: str = ""

    # ============================
    # Map Navigation
    # ============================

    def go_to_map(self, map_id: str) -> None:
        """Di chuyển đến map theo ID."""
        if self._current_map == map_id:
            logger.info(f"Navigation: already on map '{map_id}'")
            return

        map_cfg = self._loader.get_map(map_id)
        logger.info(f"Navigation: going to map '{map_cfg.name}' via {map_cfg.travel_method}")

        if map_cfg.travel_method == "teleport":
            self._teleport_to(map_cfg)
        elif map_cfg.travel_method == "walk":
            self._walk_to_map(map_cfg)
        elif map_cfg.travel_method == "minimap_click":
            self._minimap_navigate(map_cfg)
        else:
            raise BotError(
                ErrorCode.NAV_001,
                f"Unknown travel method: {map_cfg.travel_method}",
            )

        self._current_map = map_id
        logger.info(f"Navigation: arrived at '{map_cfg.name}'")

    def _teleport_to(self, map_cfg) -> None:
        """Teleport đến map (dùng UI teleport/bản đồ thế giới)."""
        if map_cfg.teleport_button:
            self._input.tap(map_cfg.teleport_button.x, map_cfg.teleport_button.y)
            time.sleep(2)
        else:
            # TODO: Mở bản đồ thế giới và tìm map
            logger.warning(f"Teleport button not configured for {map_cfg.id}")

    def _walk_to_map(self, map_cfg) -> None:
        """Di chuyển bộ (placeholder)."""
        logger.debug(f"Walk navigation to {map_cfg.id} – not fully implemented")
        time.sleep(1)

    def _minimap_navigate(self, map_cfg) -> None:
        """Click minimap để navigate."""
        logger.debug(f"Minimap navigation to {map_cfg.id} – not fully implemented")
        time.sleep(1)

    # ============================
    # Coordinate Navigation
    # ============================

    def go_to_coordinate(self, x: int, y: int, map_id: Optional[str] = None) -> None:
        """Di chuyển đến tọa độ trên màn hình."""
        if map_id:
            self.go_to_map(map_id)
        logger.info(f"Navigation: go_to_coordinate ({x}, {y})")
        self._input.tap(x, y)
        time.sleep(1)

    # ============================
    # NPC / Shop Navigation
    # ============================

    def go_to_npc(self, npc_id: str) -> None:
        """Di chuyển đến NPC."""
        npc = self._loader.get_npc(npc_id)
        logger.info(f"Navigation: going to NPC '{npc.name}' on map '{npc.map}'")
        self.go_to_map(npc.map)
        self.go_to_coordinate(npc.coordinate.x, npc.coordinate.y)

    def go_to_shop(self, shop_id: str) -> None:
        """Di chuyển đến shop thông qua NPC."""
        shops = self._loader.load_shops()
        shop = next((s for s in shops.shops if s.id == shop_id), None)
        if not shop:
            raise BotError(ErrorCode.SHOP_001, f"Shop not found: {shop_id}")
        logger.info(f"Navigation: going to shop '{shop.name}'")
        self.go_to_npc(shop.npc_id)

    def return_to_quest_giver(self, quest_giver_npc_id: str = "quest_giver_main") -> None:
        """Quay lại NPC phát nhiệm vụ."""
        logger.info(f"Navigation: returning to quest giver '{quest_giver_npc_id}'")
        self.go_to_npc(quest_giver_npc_id)

    # ============================
    # Current Position
    # ============================

    def get_current_map(self) -> str:
        return self._current_map

    def set_current_map(self, map_id: str) -> None:
        self._current_map = map_id

    def detect_current_map(self, known_maps: Optional[List[str]] = None) -> Optional[str]:
        """
        Thử detect map hiện tại qua template hoặc OCR.
        Placeholder: cần bổ sung template cho từng map.
        """
        logger.debug("NavigationEngine: detect_current_map called (basic)")
        return self._current_map or None
