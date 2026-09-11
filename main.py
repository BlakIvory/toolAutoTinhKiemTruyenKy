"""
AutoQuestTool - Entry Point CLI
Game: Tình Kiếm Truyền Kỳ

Usage:
    python main.py                    # Chạy tất cả accounts
    python main.py --check-env        # Kiểm tra môi trường
    python main.py --dry-run          # Dry run (không BUY/TRANSFER)
    python main.py --mock             # Dùng MockADB (không cần emulator)
    python main.py --account ACC_01   # Chạy một account cụ thể
    python main.py --gui              # Mở GUI (PySide6)
"""

import argparse
import sys
from pathlib import Path

# Thêm thư mục gốc vào path
sys.path.insert(0, str(Path(__file__).parent))


def run_check_env():
    """Chạy environment check."""
    import check_env
    check_env.main()


def run_bot(args):
    """Khởi động BotController."""
    # Override config từ CLI args
    if args.dry_run:
        # Patch config trước khi load
        import os
        os.environ["AUTOQUEST_DRY_RUN"] = "true"

    if args.mock:
        import os
        os.environ["AUTOQUEST_MOCK_ADB"] = "true"

    from src.bot_controller import BotController

    # Patch config nếu dry_run / mock từ CLI
    bot = BotController()

    if args.dry_run:
        bot._app_cfg.mode.dry_run = True
        print("⚠️  DRY RUN mode – BUY/TRANSFER/LOGOUT bị block")

    if args.mock:
        print("⚠️  MOCK mode – dùng MockADBController")

    if args.account:
        print(f"▶  Chạy account: {args.account}")
        bot.run_single_account(args.account)
    else:
        print("▶  Chạy tất cả accounts...")
        bot.run_all_accounts()


def run_gui():
    """Mở GUI PySide6."""
    try:
        from PySide6.QtWidgets import QApplication
        from gui.main_window import MainWindow
        import sys
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except ImportError:
        print("❌ PySide6 chưa được cài. Chạy: pip install PySide6")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="AutoQuestTool – Game Automation for Tình Kiếm Truyền Kỳ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --check-env          # Kiểm tra môi trường
  python main.py --dry-run            # Chạy không thật (không BUY/TRANSFER)
  python main.py --mock               # Test với MockADB
  python main.py --account ACC_01     # Chạy một account
  python main.py --gui                # Mở GUI
        """,
    )
    parser.add_argument(
        "--check-env", action="store_true",
        help="Kiểm tra môi trường (Python, ADB, OpenCV, ...)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Dry run – block BUY/TRANSFER/LOGOUT"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Dùng MockADB – không cần LDPlayer"
    )
    parser.add_argument(
        "--account", type=str, default="",
        help="Chỉ chạy account này (ví dụ: ACC_01)"
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Mở giao diện GUI PySide6"
    )
    parser.add_argument(
        "--config-dir", type=str, default="config",
        help="Thư mục chứa config (mặc định: config/)"
    )

    args = parser.parse_args()

    print("=" * 55)
    print("  AutoQuestTool – Tình Kiếm Truyền Kỳ Automation")
    print("=" * 55)

    if args.check_env:
        run_check_env()
        return

    if args.gui:
        run_gui()
        return

    run_bot(args)


if __name__ == "__main__":
    main()
