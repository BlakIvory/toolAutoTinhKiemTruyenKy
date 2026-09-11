"""
AutoQuestTool - Config Loader
Đọc và validate tất cả config YAML bằng Pydantic.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator
from loguru import logger

from .error_codes import ErrorCode, BotError


# ============================
# Pydantic Models
# ============================

class EmulatorConfig(BaseModel):
    type: str = "ldplayer9"
    ldplayer_path: str = ""
    ldconsole_path: str = ""
    instance_id: int = 0
    adb_host: str = "127.0.0.1"
    adb_port: int = 5555
    width: int = 1280
    height: int = 720
    dpi: int = 240


class ADBConfig(BaseModel):
    adb_path: str = ""
    connect_timeout: int = 10
    command_timeout: int = 30
    screenshot_method: str = "screencap"


class GameConfig(BaseModel):
    package_name: str
    activity_name: str = ""
    startup_wait: int = 15


class VisionConfig(BaseModel):
    default_confidence: float = 0.85
    match_method: str = "TM_CCOEFF_NORMED"


class TimingConfig(BaseModel):
    tap_delay: float = 0.3
    swipe_duration: int = 300
    screenshot_interval: float = 1.0
    action_timeout: int = 15
    login_timeout: int = 60
    navigation_timeout: int = 30


class RetryConfig(BaseModel):
    max_attempts: int = 3
    delay_between: float = 2.0


class ModeConfig(BaseModel):
    dry_run: bool = False
    mock_adb: bool = False
    debug_screenshots: bool = True
    save_path: str = "logs/debug"


class LoggingConfig(BaseModel):
    level: str = "DEBUG"
    log_dir: str = "logs"
    rotation: str = "1 day"
    retention: str = "7 days"


class AppConfig(BaseModel):
    emulator: EmulatorConfig
    adb: ADBConfig
    game: GameConfig
    vision: VisionConfig = Field(default_factory=VisionConfig)
    timing: TimingConfig = Field(default_factory=TimingConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    mode: ModeConfig = Field(default_factory=ModeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ---------------------------

class AccountConfig(BaseModel):
    id: str
    username: str
    password: str
    server: str = ""
    character: str = ""
    priority: int = 99
    enabled: bool = True
    notes: str = ""


class MasterAccountConfig(BaseModel):
    id: str
    username: str
    password: str
    server: str = ""


class AccountsConfig(BaseModel):
    accounts: List[AccountConfig] = []
    master_account: Optional[MasterAccountConfig] = None
    run_order: str = "priority"  # priority | sequential | random


# ---------------------------

class CoordinatePoint(BaseModel):
    x: int
    y: int


class CoordinateRegion(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class TemplateConfig(BaseModel):
    path: str
    confidence: float = 0.85
    region: Optional[str] = None  # Tên region trong ui_coordinates.yaml


class TemplatesConfig(BaseModel):
    templates: Dict[str, TemplateConfig] = {}


# ---------------------------

class NPCConfig(BaseModel):
    id: str
    name: str
    map: str
    coordinate: CoordinatePoint
    template: str = ""
    talk_action: str = "tap"
    interaction_region: Optional[CoordinateRegion] = None


class NPCsConfig(BaseModel):
    npcs: List[NPCConfig] = []


# ---------------------------

class MapConfig(BaseModel):
    id: str
    name: str
    map_key: str
    travel_method: str = "teleport"
    teleport_button: Optional[CoordinatePoint] = None
    npcs: List[str] = []


class MapsConfig(BaseModel):
    maps: List[MapConfig] = []
    navigation_graph: Dict[str, List[str]] = {}


# ---------------------------

class ShopItemConfig(BaseModel):
    id: str
    name: str
    quantity: int = 1
    max_spend: int = 0


class ShopConfig(BaseModel):
    id: str
    name: str
    npc_id: str
    map: str
    items: List[ShopItemConfig] = []


class ShopsConfig(BaseModel):
    shops: List[ShopConfig] = []


# ============================
# ConfigLoader
# ============================

class ConfigLoader:
    """
    Load và validate tất cả config YAML.
    Tập trung tại một class duy nhất.
    """

    CONFIG_DIR = Path("config")

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or self.CONFIG_DIR
        self._app: Optional[AppConfig] = None
        self._accounts: Optional[AccountsConfig] = None
        self._templates: Optional[TemplatesConfig] = None
        self._maps: Optional[MapsConfig] = None
        self._npcs: Optional[NPCsConfig] = None
        self._shops: Optional[ShopsConfig] = None
        self._ui_coords: Optional[Dict[str, Any]] = None

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _load_yaml(self, filename: str) -> dict:
        path = self.config_dir / filename
        if not path.exists():
            raise BotError(
                ErrorCode.CFG_001,
                f"Config file not found: {path}",
            )
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f) or {}
                return data
            except yaml.YAMLError as e:
                raise BotError(
                    ErrorCode.CFG_002,
                    f"YAML parse error in {filename}: {e}",
                )

    def _validate(self, model_cls, data: dict, filename: str):
        try:
            return model_cls(**data)
        except Exception as e:
            raise BotError(
                ErrorCode.CFG_003,
                f"Validation error in {filename}: {e}",
            )

    # ---------------------------
    # Public loaders
    # ---------------------------

    def load_app(self) -> AppConfig:
        if self._app is None:
            data = self._load_yaml("app.yaml")
            self._app = self._validate(AppConfig, data, "app.yaml")
            logger.info("Loaded app.yaml OK")
        return self._app

    def load_accounts(self) -> AccountsConfig:
        if self._accounts is None:
            data = self._load_yaml("accounts.yaml")
            self._accounts = self._validate(AccountsConfig, data, "accounts.yaml")
            enabled = [a for a in self._accounts.accounts if a.enabled]
            logger.info(f"Loaded accounts.yaml: {len(enabled)} account(s) enabled")
        return self._accounts

    def load_templates(self) -> TemplatesConfig:
        if self._templates is None:
            data = self._load_yaml("templates.yaml")
            # templates.yaml có key 'templates' → convert
            templates_raw = data.get("templates", {})
            self._templates = TemplatesConfig(templates={
                k: TemplateConfig(**v) for k, v in templates_raw.items()
            })
            logger.info(f"Loaded templates.yaml: {len(templates_raw)} template(s)")
        return self._templates

    def load_maps(self) -> MapsConfig:
        if self._maps is None:
            data = self._load_yaml("maps.yaml")
            self._maps = self._validate(MapsConfig, data, "maps.yaml")
            logger.info(f"Loaded maps.yaml: {len(self._maps.maps)} map(s)")
        return self._maps

    def load_npcs(self) -> NPCsConfig:
        if self._npcs is None:
            data = self._load_yaml("npcs.yaml")
            self._npcs = self._validate(NPCsConfig, data, "npcs.yaml")
            logger.info(f"Loaded npcs.yaml: {len(self._npcs.npcs)} NPC(s)")
        return self._npcs

    def load_shops(self) -> ShopsConfig:
        if self._shops is None:
            data = self._load_yaml("shops.yaml")
            self._shops = self._validate(ShopsConfig, data, "shops.yaml")
            logger.info(f"Loaded shops.yaml: {len(self._shops.shops)} shop(s)")
        return self._shops

    def load_ui_coords(self) -> Dict[str, Any]:
        if self._ui_coords is None:
            self._ui_coords = self._load_yaml("ui_coordinates.yaml")
            logger.info("Loaded ui_coordinates.yaml OK")
        return self._ui_coords

    def load_all(self) -> None:
        """Load tất cả config một lần khi startup."""
        self.load_app()
        self.load_accounts()
        self.load_templates()
        self.load_maps()
        self.load_npcs()
        self.load_shops()
        self.load_ui_coords()
        logger.info("All configs loaded successfully.")

    # ---------------------------
    # Helpers
    # ---------------------------

    def get_enabled_accounts(self) -> List[AccountConfig]:
        cfg = self.load_accounts()
        accounts = [a for a in cfg.accounts if a.enabled]
        if cfg.run_order == "priority":
            accounts.sort(key=lambda a: a.priority)
        return accounts

    def get_template(self, name: str) -> TemplateConfig:
        cfg = self.load_templates()
        if name not in cfg.templates:
            raise BotError(ErrorCode.VIS_002, f"Template not in config: {name}")
        return cfg.templates[name]

    def get_npc(self, npc_id: str) -> NPCConfig:
        cfg = self.load_npcs()
        for npc in cfg.npcs:
            if npc.id == npc_id:
                return npc
        raise BotError(ErrorCode.NPC_001, f"NPC not found in config: {npc_id}")

    def get_map(self, map_id: str) -> MapConfig:
        cfg = self.load_maps()
        for m in cfg.maps:
            if m.id == map_id:
                return m
        raise BotError(ErrorCode.NAV_001, f"Map not found in config: {map_id}")

    def get_region(self, region_name: str) -> Tuple[int, int, int, int]:
        """Trả về (x1, y1, x2, y2) của region."""
        coords = self.load_ui_coords()
        regions = coords.get("regions", {})
        if region_name not in regions:
            raise BotError(ErrorCode.CFG_004, f"Region not found: {region_name}")
        r = regions[region_name]
        return (r["x1"], r["y1"], r["x2"], r["y2"])
