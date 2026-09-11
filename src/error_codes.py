"""
AutoQuestTool - Error Codes
Tất cả error codes được định nghĩa tập trung ở đây.
"""

from enum import Enum


class ErrorCode(str, Enum):
    """Error codes theo module."""

    # --- ADB ---
    ADB_001 = "ADB_001"   # ADB không tìm thấy / không kết nối được
    ADB_002 = "ADB_002"   # Không có device
    ADB_003 = "ADB_003"   # Screenshot thất bại
    ADB_004 = "ADB_004"   # Tap/swipe thất bại
    ADB_005 = "ADB_005"   # Input text thất bại
    ADB_006 = "ADB_006"   # Command timeout

    # --- Config ---
    CFG_001 = "CFG_001"   # File config không tìm thấy
    CFG_002 = "CFG_002"   # Config parse error
    CFG_003 = "CFG_003"   # Validation error (Pydantic)
    CFG_004 = "CFG_004"   # Thiếu field bắt buộc

    # --- Vision ---
    VIS_001 = "VIS_001"   # Template không tìm thấy trên màn hình
    VIS_002 = "VIS_002"   # Template file không tồn tại
    VIS_003 = "VIS_003"   # Screenshot null / không hợp lệ
    VIS_004 = "VIS_004"   # Confidence dưới threshold

    # --- Login ---
    LOGIN_001 = "LOGIN_001"  # Không phát hiện màn hình login
    LOGIN_002 = "LOGIN_002"  # Login thất bại sau max_attempts
    LOGIN_003 = "LOGIN_003"  # Không phát hiện in-game sau login
    LOGIN_004 = "LOGIN_004"  # Username/password không hợp lệ

    # --- Navigation ---
    NAV_001 = "NAV_001"   # Không tìm thấy map đích
    NAV_002 = "NAV_002"   # Navigation timeout
    NAV_003 = "NAV_003"   # Unknown map / không xác định được map hiện tại

    # --- NPC ---
    NPC_001 = "NPC_001"   # Không tìm thấy NPC
    NPC_002 = "NPC_002"   # NPC không phản hồi
    NPC_003 = "NPC_003"   # Interact NPC thất bại

    # --- Shop ---
    SHOP_001 = "SHOP_001"  # Không mở được shop
    SHOP_002 = "SHOP_002"  # Item không có trong shop
    SHOP_003 = "SHOP_003"  # Mua thất bại
    SHOP_004 = "SHOP_004"  # Không đủ vàng

    # --- Quest ---
    QUEST_001 = "QUEST_001"  # Không có nhiệm vụ
    QUEST_002 = "QUEST_002"  # Classify UNKNOWN
    QUEST_003 = "QUEST_003"  # Accept quest thất bại
    QUEST_004 = "QUEST_004"  # Complete quest thất bại
    QUEST_005 = "QUEST_005"  # Cancel quest thất bại

    # --- Transfer ---
    TRANSFER_001 = "TRANSFER_001"  # Transfer gold thất bại
    TRANSFER_002 = "TRANSFER_002"  # Không đủ vàng để transfer
    TRANSFER_003 = "TRANSFER_003"  # Blocked bởi DRY_RUN

    # --- Watchdog ---
    WD_001 = "WD_001"   # Game crash detected
    WD_002 = "WD_002"   # Game freeze detected
    WD_003 = "WD_003"   # Loading timeout
    WD_004 = "WD_004"   # Unknown screen
    WD_005 = "WD_005"   # ADB disconnected
    WD_006 = "WD_006"   # State timeout

    # --- State Machine ---
    SM_001 = "SM_001"   # Invalid state transition
    SM_002 = "SM_002"   # Max retry vượt quá
    SM_003 = "SM_003"   # State timeout


class BotError(Exception):
    """Exception tùy chỉnh cho AutoQuestTool."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        account: str = "",
        state: str = "",
        action: str = "",
        screenshot_path: str = "",
        retry_count: int = 0,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.account = account
        self.state = state
        self.action = action
        self.screenshot_path = screenshot_path
        self.retry_count = retry_count

    def __repr__(self) -> str:
        return (
            f"BotError(code={self.code}, "
            f"account={self.account!r}, "
            f"state={self.state!r}, "
            f"msg={self.message!r})"
        )

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "message": self.message,
            "account": self.account,
            "state": self.state,
            "action": self.action,
            "screenshot_path": self.screenshot_path,
            "retry_count": self.retry_count,
        }
