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

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Thêm thư mục gốc vào path
sys.path.insert(0, str(Path(__file__).parent))


def run_check_env():
    """Chay environment check."""
    import check_env
    check_env.main()


def run_launcher():
    """Mo Launcher GUI chon may ao va khoi dong bot."""
    from launcher import main as launcher_main
    launcher_main()


def run_bot(args):
    """Khởi động BotController."""
    # Override config từ CLI args
    if args.dry_run:
        import os
        os.environ["AUTOQUEST_DRY_RUN"] = "true"

    if args.mock:
        import os
        os.environ["AUTOQUEST_MOCK_ADB"] = "true"

    from src.bot_controller import BotController
    from src.state_machine import BotState

    bot = BotController()

    if args.dry_run:
        bot._app_cfg.mode.dry_run = True
        print("[DRY RUN] BUY/TRANSFER/LOGOUT bi block")

    if args.mock:
        print("[MOCK] Dung MockADBController")

    if getattr(args, "device", None):
        bot._adb.set_target_device(args.device)
        print(f"[DEVICE] Target ADB serial: {args.device}")

    if getattr(args, "instance", None) is not None:
        bot._target_instance_index = int(args.instance)
        print(f"[INSTANCE] LDPlayer instance index: {args.instance}")

    if getattr(args, "server", None):
        bot._target_server = args.server
        print(f"[SERVER] Game server: {args.server}")

    # --skip-login: bo qua man hinh login (neu da vao game roi)
    # Bat dau tu START_GAME: tu kiem tra game co mo khong, mo neu chua co
    # Neu da vao game san → skip login, vao thang GET_QUEST
    initial_state = BotState.BOOT
    if args.skip_login:
        bot._skip_login = True
        initial_state = BotState.FIND_COMMERCE_CHIEF
        print("[SKIP LOGIN] Bat dau tim Chu thuong hoi (game dang chay san)")

    if getattr(args, "accounts", ""):
        account_ids = [value.strip() for value in args.accounts.split(",") if value.strip()]
        print(f"[ACCOUNTS] Hang doi: {', '.join(account_ids)}")
        bot.run_all_accounts(account_ids)
    elif args.account:
        print(f"[ACCOUNT] Chay: {args.account}")
        bot._state_machine.run(account_id=args.account, initial_state=initial_state)
    else:
        if initial_state != BotState.BOOT:
            # Skip login: chạy account đầu tiên từ state chỉ định
            accounts = bot._loader.get_enabled_accounts()
            if accounts:
                print(f"[SKIP LOGIN] Chay account: {accounts[0].id}")
                bot._state_machine.run(account_id=accounts[0].id, initial_state=initial_state)
            else:
                print("[ERROR] Khong co account nao duoc enable trong config!")
        else:
            print("[START] Chay tat ca accounts...")
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
        help="Dung MockADB – khong can LDPlayer"
    )
    parser.add_argument(
        "--skip-login", action="store_true",
        help="Bo qua Login, bat dau tu nhiem vu Thuong hoi"
    )
    parser.add_argument(
        "--account", type=str, default="",
        help="Chi chay account nay (vi du: ACC_01)"
    )
    parser.add_argument(
        "--accounts", type=str, default="",
        help="Hang doi account cach nhau bang dau phay (vi du: ACC_01,ACC_03)"
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Mo giao dien GUI PySide6"
    )
    parser.add_argument(
        "--config-dir", type=str, default="config",
        help="Thu muc chua config (mac dinh: config/)"
    )

    parser.add_argument(
        "--launcher", action="store_true",
        help="Mo Launcher GUI de chon may ao LDPlayer va khoi dong bot"
    )
    parser.add_argument(
        "--server", type=str, default="S1",
        help="Server game (vi du: S1, S2)"
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="Target ADB serial (vi du: 127.0.0.1:5555 hoac emulator-5556)"
    )
    parser.add_argument(
        "--instance", type=int, default=None,
        help="LDPlayer instance index (vi du: 0, 1)"
    )

    args = parser.parse_args()

    print("=" * 55)
    print("  AutoQuestTool - Tinh Kiem Truyen Ky Automation")
    print("=" * 55)

    if args.check_env:
        run_check_env()
        return

    if args.launcher:
        run_launcher()
        return

    if args.gui:
        run_gui()
        return

    run_bot(args)


if __name__ == "__main__":
    main()
