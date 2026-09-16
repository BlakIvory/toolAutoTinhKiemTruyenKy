"""
AutoQuestTool - Launcher GUI (v2)
- Tab 1: Máy ảo LDPlayer (chọn instance, mở game)
- Tab 2: Tài khoản (thêm/sửa/xóa account, lưu vào accounts.yaml)

Chạy: python launcher.py
"""

from __future__ import annotations

import os
import sys
import re
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.config_loader import ConfigLoader
from src.ldplayer_controller import LDPlayerController, LDInstance, GAME_PACKAGE

# ── Màu sắc & style ───────────────────────────────────────────
BG       = "#1a1a2e"
BG2      = "#16213e"
BG3      = "#0d1117"
ACCENT   = "#0f3460"
ACCENT2  = "#1a4a80"
GREEN    = "#4ade80"
RED      = "#f87171"
YELLOW   = "#fbbf24"
WHITE    = "#f1f5f9"
GRAY     = "#64748b"
GRAY2    = "#374151"
FONT     = ("Segoe UI", 10)
FONT_B   = ("Segoe UI", 10, "bold")
FONT_H   = ("Segoe UI", 13, "bold")
FONT_SM  = ("Segoe UI", 9)
FONT_C   = ("Consolas", 9)

ACCOUNTS_FILE = Path(__file__).parent / "config" / "accounts.yaml"


# ══════════════════════════════════════════════════════════════
# Dialog thêm / sửa Account
# ══════════════════════════════════════════════════════════════

class AccountDialog(tk.Toplevel):
    """Cửa sổ nhập thông tin 1 account."""

    def __init__(self, parent, title="Thêm tài khoản", data: dict | None = None, readonly_id: bool = False):
        super().__init__(parent)
        self.title(title)
        self.configure(bg=BG)
        self.resizable(False, False)
        self.grab_set()  # Modal
        self.result: dict | None = None

        fields = [
            ("ID (tự động)",              "id"),
            ("Username / Tên đăng nhập",  "username"),
            ("Password",                  "password"),
            ("Ghi chú",                   "notes"),
        ]

        self._vars: dict[str, tk.StringVar] = {}
        for i, (label, key) in enumerate(fields):
            tk.Label(self, text=label, font=FONT_SM, fg=GRAY, bg=BG,
                     anchor="w").grid(row=i, column=0, padx=16, pady=4, sticky="w")
            var = tk.StringVar(value=(data or {}).get(key, ""))
            show = "*" if key == "password" else ""
            # ID read-only khi readonly_id=True (thêm mới, ID đã tự sinh)
            state = "readonly" if (key == "id" and readonly_id) else "normal"
            entry = tk.Entry(self, textvariable=var, font=FONT,
                             bg=BG2 if state == "normal" else ACCENT,
                             fg=WHITE, insertbackground=WHITE,
                             disabledforeground=GRAY,
                             relief="flat", width=28, show=show,
                             state=state)
            entry.grid(row=i, column=1, padx=16, pady=4, sticky="ew")
            self._vars[key] = var

        # Enabled checkbox
        self._enabled = tk.BooleanVar(value=(data or {}).get("enabled", True))
        tk.Checkbutton(self, text="Bật (enabled)", variable=self._enabled,
                       font=FONT, fg=WHITE, bg=BG, activebackground=BG,
                       selectcolor=ACCENT).grid(
            row=len(fields), column=0, columnspan=2, pady=6, padx=16, sticky="w")

        # Buttons
        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.grid(row=len(fields)+1, column=0, columnspan=2, pady=12)
        tk.Button(btn_frame, text="💾 Lưu", font=FONT_B,
                  bg=GREEN, fg="#0a0a0a", relief="flat", padx=16, cursor="hand2",
                  command=self._save).pack(side="left", padx=8)
        tk.Button(btn_frame, text="✗ Huỷ", font=FONT,
                  bg=GRAY2, fg=WHITE, relief="flat", padx=16, cursor="hand2",
                  command=self.destroy).pack(side="left")

        self._center()

    def _center(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        x = (self.winfo_screenwidth() - 400) // 2
        y = (self.winfo_screenheight() - 300) // 2
        self.geometry(f"400x300+{x}+{y}")

    def _save(self):
        acc_id = self._vars["id"].get().strip()
        username = self._vars["username"].get().strip()
        if not acc_id or not username:
            messagebox.showwarning("Thiếu thông tin", "ID và Username không được trống!", parent=self)
            return
        self.result = {
            "id":        acc_id,
            "username":  username,
            "password":  self._vars["password"].get(),
            "server":    "S1",
            "priority":  1,
            "enabled":   self._enabled.get(),
            "notes":     self._vars["notes"].get().strip(),
        }
        self.destroy()


# ══════════════════════════════════════════════════════════════
# Tab 1: Máy ảo LDPlayer
# ══════════════════════════════════════════════════════════════

class InstanceRow(tk.Frame):
    def __init__(self, parent, instance: LDInstance, **kw):
        super().__init__(parent, bg=BG2, pady=6, padx=10, **kw)
        self.instance = instance
        self.var = tk.BooleanVar(value=True)

        tk.Checkbutton(self, variable=self.var, bg=BG2, activebackground=BG2,
                       fg=WHITE, activeforeground=WHITE,
                       selectcolor=ACCENT, cursor="hand2").pack(side="left")
        tk.Label(self, text=f"  {instance.name}", font=FONT_B,
                 fg=WHITE, bg=BG2, width=14, anchor="w").pack(side="left")
        tk.Label(self, text=f"[Index {instance.index}]", font=FONT_SM,
                 fg=GRAY, bg=BG2, width=10, anchor="w").pack(side="left")
        acc_label = f"ACC_{instance.index + 1:02d}"
        tk.Label(self, text=f"({acc_label})", font=FONT_SM,
                 fg="#38bdf8", bg=BG2, width=10, anchor="w").pack(side="left")

        color = GREEN if instance.is_running else RED
        status = "● Đang chạy" if instance.is_running else "○ Đã tắt"
        self.status_lbl = tk.Label(self, text=status, font=FONT_SM,
                                   fg=color, bg=BG2, width=12, anchor="w")
        self.status_lbl.pack(side="left")

        tk.Frame(self, bg="#2d3748", height=1).pack(fill="x", side="bottom")

    def update_status(self, is_running: bool):
        self.instance.is_running = is_running
        color = GREEN if is_running else RED
        self.status_lbl.config(text="● Đang chạy" if is_running else "○ Đã tắt", fg=color)

    @property
    def selected(self) -> bool:
        return self.var.get()


class EmulatorTab(tk.Frame):
    def __init__(self, parent, ld: LDPlayerController, log_fn, app_cfg=None, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._ld = ld
        self._log = log_fn
        self._app_cfg = app_cfg
        self._rows: list[InstanceRow] = []
        self._bot_processes: list = []
        self._is_running = False
        self._build()
        # Defer so parent log box is fully built before first log call
        self.after(100, self._refresh)

    def _build(self):
        toolbar = tk.Frame(self, bg=BG, pady=6, padx=12)
        toolbar.pack(fill="x")
        tk.Label(toolbar, text="Danh sách máy ảo LDPlayer:",
                 font=FONT_B, fg=WHITE, bg=BG).pack(side="left")
        tk.Button(toolbar, text="🔄 Refresh", font=FONT, bg=ACCENT, fg=WHITE,
                  activebackground=ACCENT2, activeforeground=WHITE,
                  relief="flat", padx=10, cursor="hand2",
                  command=self._refresh).pack(side="right")

        container = tk.Frame(self, bg=BG2)
        container.pack(fill="both", expand=True, padx=12, pady=4)

        canvas = tk.Canvas(container, bg=BG2, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self._list_frame = tk.Frame(canvas, bg=BG2)

        self._list_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas_window = canvas.create_window((0, 0), window=self._list_frame, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        self._list_frame.bind("<MouseWheel>", _on_mousewheel)

        btn_frame = tk.Frame(self, bg=BG, pady=8, padx=12)
        btn_frame.pack(fill="x")
        tk.Button(btn_frame, text="✓ Chọn tất cả", font=FONT, bg=ACCENT, fg=WHITE,
                  activebackground=ACCENT2, activeforeground=WHITE,
                  relief="flat", padx=12, cursor="hand2",
                  command=self._select_all).pack(side="left", padx=(0, 6))
        tk.Button(btn_frame, text="✗ Bỏ chọn", font=FONT, bg=GRAY2, fg=WHITE,
                  activebackground="#4b5563", activeforeground=WHITE,
                  relief="flat", padx=12, cursor="hand2",
                  command=self._deselect_all).pack(side="left", padx=(0, 6))

        # Nút Start / End và Server: hiển thị theo thứ tự Server: -> dropdown -> Start/End
        self._start_btn = tk.Button(
            btn_frame, text="▶  Start Bot",
            font=("Segoe UI", 11, "bold"), bg=GREEN, fg="#0a0a0a",
            activebackground="#22c55e", activeforeground="#0a0a0a",
            relief="flat", padx=20, pady=4, cursor="hand2",
            command=self._toggle_bot)
        self._start_btn.pack(side="right", padx=(8, 0))

        self._server_var = tk.StringVar(value="S1")
        server_cb = ttk.Combobox(
            btn_frame, textvariable=self._server_var,
            values=[f"S{i}" for i in range(1, 21)],
            width=5, state="readonly", font=FONT
        )
        server_cb.pack(side="right", padx=(4, 0))

        tk.Label(btn_frame, text="Server:", font=FONT_B,
                 fg=WHITE, bg=BG).pack(side="right", padx=(12, 2))

    def _refresh(self):
        for w in self._list_frame.winfo_children():
            w.destroy()
        self._rows.clear()

        if not self._ld.available:
            tk.Label(self._list_frame,
                     text="⚠  Không tìm thấy ldconsole.exe!\nKiểm tra config/app.yaml",
                     font=FONT, fg=RED, bg=BG2, justify="center").pack(pady=30)
            self._log("⚠ ldconsole.exe không tìm thấy!")
            return

        instances = self._ld.list_instances()
        if not instances:
            tk.Label(self._list_frame,
                     text="Không có instance nào. Mở LDPlayer rồi bấm Refresh.",
                     font=FONT, fg=GRAY, bg=BG2, justify="center").pack(pady=30)
            return

        for inst in instances:
            if inst.index >= 99999 or inst.index < 0 or not inst.name.strip():
                continue
            row = InstanceRow(self._list_frame, inst)
            row.pack(fill="x", pady=2)
            self._rows.append(row)
        self._log(f"✅ Tìm thấy {len(instances)} instance(s)")

    def _select_all(self):
        for r in self._rows: r.var.set(True)

    def _deselect_all(self):
        for r in self._rows: r.var.set(False)

    def _set_button_start(self):
        self._start_btn.config(
            text="▶  Start Bot",
            bg=GREEN, fg="#0a0a0a",
            activebackground="#22c55e", activeforeground="#0a0a0a",
            state="normal", cursor="hand2"
        )
        self._is_running = False

    def _set_button_end(self):
        self._start_btn.config(
            text="⏹  End",
            bg=RED, fg=WHITE,
            activebackground="#ef4444", activeforeground=WHITE,
            state="normal", cursor="hand2"
        )
        self._is_running = True

    def _toggle_bot(self):
        if self._is_running:
            self._stop_bot()
        else:
            self._start_bot()

    def _stop_bot(self):
        self._start_btn.config(state="disabled", text="⏳ Đang dừng...")
        threading.Thread(target=self._stop_thread, daemon=True).start()

    def _cleanup_old_bot_processes(self):
        """Dừng sạch các tiến trình main.py cũ để tránh chạy song song xung đột."""
        count = 0
        for proc in list(self._bot_processes):
            try:
                import subprocess
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True, timeout=5)
                count += 1
            except Exception:
                try:
                    proc.kill()
                    count += 1
                except Exception:
                    pass
        self._bot_processes.clear()

        try:
            import subprocess
            flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            ps_cmd = (
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.CommandLine -like '*main.py*' -and $_.CommandLine -notlike '*launcher.py*' } | "
                "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
            )
            subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd],
                           capture_output=True, timeout=5, creationflags=flags)
        except Exception:
            pass

        return count

    def _stop_thread(self):
        self._log("\n⏹  Đang dừng toàn bộ tool và tắt các tab máy ảo...")

        # 1. Dừng các tiến trình bot (main.py)
        count = self._cleanup_old_bot_processes()
        if count > 0:
            self._log(f"  ✅ Đã dừng {count} tiến trình bot.")

        # 2. Tắt các tab máy ảo đã chọn
        for row in self._rows:
            if row.selected:
                inst = row.instance
                self._log(f"  ⏳ Đang tắt tab máy ảo [{inst.name}] (Index {inst.index})...")
                try:
                    self._ld.stop_instance(inst.index)
                    self.after(0, lambda r=row: r.update_status(False))
                    self._log(f"  ✅ Đã tắt tab [{inst.name}]")
                except Exception as e:
                    self._log(f"  ❌ Lỗi khi tắt tab [{inst.name}]: {e}")

        self._log("🎉 Đã dừng toàn bộ bot và tắt các tab máy ảo thành công!")
        self.after(0, self._set_button_start)

    def _start_bot(self):
        selected = [r for r in self._rows if r.selected]
        if not selected:
            messagebox.showwarning("Chưa chọn máy",
                                   "Vui lòng chọn ít nhất 1 máy ảo!")
            return
        self._start_btn.config(state="disabled", text="⏳ Đang khởi động...")
        self._log(f"▶ Khởi động {len(selected)} máy...")
        threading.Thread(target=self._run_thread, args=(selected,), daemon=True).start()

    def _run_thread(self, rows: list[InstanceRow]):
        import subprocess

        # Dọn dẹp sạch mọi tiến trình bot cũ trước khi khởi chạy
        self._cleanup_old_bot_processes()

        try:
            account_data = yaml.safe_load(ACCOUNTS_FILE.read_text(encoding="utf-8")) or {}
            enabled_accounts = [a for a in account_data.get("accounts", []) if a.get("enabled", True)]
            enabled_accounts.sort(key=lambda a: a.get("priority", 99))
        except Exception as exc:
            self._log(f"Could not load account queue: {exc}")
            enabled_accounts = []
        account_queues = [enabled_accounts[i::len(rows)] for i in range(len(rows))]

        for i, row in enumerate(rows):
            inst = row.instance
            self._log(f"\n📱 [{inst.name}] (Index {inst.index}) — Đang kiểm tra...")

            # 1. KIỂM TRA TAB MÁY ẢO ĐÃ MỞ CHƯA
            self._log(f"  🔍 Kiểm tra tab máy ảo...")
            if self._ld.is_instance_running(inst.index):
                self._log(f"  ✅ Tab [{inst.name}] ĐÃ MỞ SẴN (đang chạy).")
                self.after(0, lambda r=row: r.update_status(True))
            else:
                self._log(f"  ⏳ Tab [{inst.name}] CHƯA MỞ. Đang khởi động tab...")
                try:
                    self._ld.launch_instance(inst.index, wait=2)
                except Exception as e:
                    self._log(f"  ❌ Lỗi lệnh mở tab: {e}")
                    continue

                # Polling kiểm tra tab đã mở thành công chưa
                self._log(f"  ⏳ Đang chờ tab [{inst.name}] load xong...")
                tab_ready = False
                for sec in range(1, 35):
                    time.sleep(1)
                    if self._ld.is_instance_running(inst.index):
                        tab_ready = True
                        break
                    if sec % 5 == 0:
                        self._log(f"    ...vẫn đang chờ tab mở ({sec}s)")

                if not tab_ready:
                    self._log(f"  ❌ Quá thời gian chờ mở tab [{inst.name}]! Bỏ qua máy này.")
                    continue

                self._log(f"  ✅ Tab [{inst.name}] đã mở thành công!")
                self.after(0, lambda r=row: r.update_status(True))
                time.sleep(1)  # Các bước sau vẫn polling trạng thái

            # 2. KIỂM TRA GAME TRONG TAB ĐÃ MỞ CHƯA
            self._log(f"  🔍 Kiểm tra game trong tab [{inst.name}]...")
            if self._ld.is_app_running(inst.index, GAME_PACKAGE):
                self._log(f"  ✅ Game ĐÃ MỞ SẴN trong tab! Bỏ qua thời gian chờ.")
            else:
                self._log(f"  ⏳ Game chưa mở trong tab. Đang mở game...")
                try:
                    self._ld.launch_app(inst.index, GAME_PACKAGE, wait=3)
                except Exception as e:
                    self._log(f"  ❌ Lỗi mở game: {e}")
                    continue

                # Polling kiểm tra game đã chạy chưa
                self._log(f"  ⏳ Đang chờ game khởi động...")
                game_ready = False
                for sec in range(1, 25):
                    time.sleep(1)
                    if self._ld.is_app_running(inst.index, GAME_PACKAGE):
                        game_ready = True
                        break
                    if sec % 5 == 0:
                        self._log(f"    ...đang chờ game load ({sec}s)")

                if game_ready:
                    self._log(f"  ✅ Game đã sẵn sàng trong tab!")
                else:
                    self._log(f"  ⚠ Game đang load, tiếp tục khởi động bot...")

            # 3. KHỞI ĐỘNG BOT CHO TAB NÀY
            tcp_port = 5555 + inst.index * 2
            target_device = f"127.0.0.1:{tcp_port}"
            try:
                adb_bin = getattr(self, "_app_cfg", None) and self._app_cfg.adb.adb_path or "adb"
                flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                self._log(f"  🔌 Đang kết nối ADB tới tab [{inst.name}] ({target_device})...")
                for _ in range(10):
                    subprocess.run([adb_bin, "connect", target_device], capture_output=True, timeout=3, creationflags=flags)
                    chk = subprocess.run([adb_bin, "-s", target_device, "get-state"], capture_output=True, text=True, timeout=2, creationflags=flags)
                    if "device" in chk.stdout.lower():
                        self._log(f"  ✅ ADB kết nối thành công tới [{inst.name}]!")
                        break
                    time.sleep(1.5)
            except Exception as e:
                self._log(f"  ⚠️ Lỗi connect ADB: {e}")

            server = self._server_var.get()
            queue = account_queues[i]
            if not queue:
                self._log(f"  Không còn tài khoản chưa chạy cho tab [{inst.name}]")
                continue
            account_ids = [account["id"] for account in queue]
            self._log(f"  🤖 Hàng đợi: {', '.join(account_ids)} | Server: {server} | Thiết bị: {target_device}...")

            cmd = [
                sys.executable, "main.py",
                "--accounts", ",".join(account_ids),
                "--server", server,
                "--device", target_device,
                "--instance", str(inst.index),
            ]
            try:
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                proc = subprocess.Popen(
                    cmd,
                    creationflags=(
                        subprocess.CREATE_NO_WINDOW
                        if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                    ),
                    cwd=str(Path(__file__).parent),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                self._bot_processes.append(proc)
                self._log(f"  ✅ Bot cho tab [{inst.name}] đã chạy (PID {proc.pid})!")

                def _stream_output(p, name):
                    try:
                        for line in p.stdout:
                            l = line.strip()
                            if not l:
                                continue
                            if any(kw in l for kw in ["[ACC_", "ADB", "Dang dang nhap", "Logging in", "Bắt đầu", "Kết thúc", "BotError", "Error", "ERROR", "Vào game", "Chọn nhân vật", "WARNING", "SUCCESS", "Thành công"]):
                                self.after(0, lambda msg=f"  [{name}] {l}": self._log(msg))
                    except Exception:
                        pass

                threading.Thread(target=_stream_output, args=(proc, inst.name), daemon=True).start()
            except Exception as e:
                self._log(f"  ❌ Lỗi khởi động bot: {e}")

            # 4. TRƯỚC KHI START TIẾP MÁY TIẾP THEO
            if i < len(rows) - 1:
                next_inst = rows[i + 1].instance
                self._log(f"  ⏳ Đã hoàn thành tab [{inst.name}]. Đang start tab [{next_inst.name}]...")
                time.sleep(0.2)

        self._log("\n🎉 Đã hoàn tất kiểm tra và khởi động tất cả tab đã chọn!")
        if self._bot_processes:
            self.after(0, self._set_button_end)
        else:
            self.after(0, self._set_button_start)


# ══════════════════════════════════════════════════════════════
# Tab 2: Tài khoản
# ══════════════════════════════════════════════════════════════

class AccountsTab(tk.Frame):
    def __init__(self, parent, log_fn, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._log = log_fn
        self._accounts: list[dict] = []
        self._meta: dict = {}
        self._build()
        # Defer refresh so parent log box is ready
        self.after(150, self._load)

    def _build(self):
        # Toolbar
        toolbar = tk.Frame(self, bg=BG, pady=6, padx=12)
        toolbar.pack(fill="x")
        tk.Label(toolbar, text="Danh sách tài khoản đăng nhập:",
                 font=FONT_B, fg=WHITE, bg=BG).pack(side="left")

        # Bảng tài khoản (Treeview)
        table_frame = tk.Frame(self, bg=BG2, padx=8, pady=8)
        table_frame.pack(fill="both", expand=True, padx=12, pady=4)

        cols = ("id", "username", "password", "server", "enabled", "notes")
        col_labels = ("ID", "Username", "Password", "Server", "Bật", "Ghi chú")
        col_widths = (70, 130, 100, 60, 50, 140)

        self._tree = ttk.Treeview(
            table_frame, columns=cols, show="headings",
            height=8, selectmode="browse"
        )
        for col, label, w in zip(cols, col_labels, col_widths):
            self._tree.heading(col, text=label, anchor="w")
            self._tree.column(col, width=w, minwidth=w, anchor="w")

        # Style treeview
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",
            background=BG2, foreground=WHITE,
            fieldbackground=BG2, rowheight=26, font=FONT_SM)
        style.configure("Treeview.Heading",
            background=ACCENT, foreground=WHITE, font=FONT_B, relief="flat")
        style.map("Treeview", background=[("selected", ACCENT2)])

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical",
                                   command=self._tree.yview)
        self._tree.configure(yscrollcommand=scrollbar.set)
        self._tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self._tree.bind("<Double-1>", lambda e: self._edit_account())

        # Buttons
        btn_frame = tk.Frame(self, bg=BG, pady=8, padx=12)
        btn_frame.pack(fill="x")

        for text, cmd, color in [
            ("➕ Thêm TK",  self._add_account,    ACCENT),
            ("✏  Sửa TK",   self._edit_account,   GRAY2),
            ("🗑  Xóa TK",   self._delete_account, "#7f1d1d"),
        ]:
            tk.Button(btn_frame, text=text, font=FONT, bg=color, fg=WHITE,
                      activebackground=ACCENT2, activeforeground=WHITE,
                      relief="flat", padx=12, cursor="hand2",
                      command=cmd).pack(side="left", padx=(0, 6))

        tk.Button(btn_frame, text="💾 Lưu vào file",
                  font=("Segoe UI", 10, "bold"), bg=GREEN, fg="#0a0a0a",
                  activebackground="#22c55e", activeforeground="#0a0a0a",
                  relief="flat", padx=16, cursor="hand2",
                  command=self._save).pack(side="right")

        # Hint
        tk.Label(self, text="💡 Double-click để sửa nhanh  |  Nhấn 'Lưu vào file' để lưu thay đổi",
                 font=FONT_SM, fg=GRAY, bg=BG).pack(pady=(0, 6))

    # ── Data ──────────────────────────────────────────────────

    def _load(self):
        """Đọc accounts.yaml."""
        try:
            with open(ACCOUNTS_FILE, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self._accounts = data.get("accounts", [])
            self._meta = {k: v for k, v in data.items() if k != "accounts"}
            self._refresh_table()
            self._log(f"📂 Đã tải {len(self._accounts)} tài khoản từ accounts.yaml")
        except Exception as e:
            self._log(f"⚠ Không đọc được accounts.yaml: {e}")

    def _save(self):
        """Ghi accounts.yaml."""
        try:
            data = dict(self._meta)
            data["accounts"] = self._accounts
            with open(ACCOUNTS_FILE, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True,
                          default_flow_style=False, sort_keys=False)
            self._log(f"💾 Đã lưu {len(self._accounts)} tài khoản vào accounts.yaml")
            messagebox.showinfo("Đã lưu",
                                f"Lưu {len(self._accounts)} tài khoản thành công!")
        except Exception as e:
            self._log(f"❌ Lỗi lưu: {e}")
            messagebox.showerror("Lỗi", f"Không lưu được: {e}")

    def _refresh_table(self):
        for item in self._tree.get_children():
            self._tree.delete(item)
        for acc in self._accounts:
            pwd_masked = "●" * min(len(acc.get("password", "")), 8)
            enabled_str = "✅" if acc.get("enabled", True) else "❌"
            self._tree.insert("", "end", values=(
                acc.get("id", ""),
                acc.get("username", ""),
                pwd_masked,
                acc.get("server", "S1"),
                enabled_str,
                acc.get("notes", ""),
            ))

    def _selected_index(self) -> int | None:
        sel = self._tree.selection()
        if not sel:
            return None
        return self._tree.index(sel[0])

    # ── CRUD ──────────────────────────────────────────────────

    def _next_id(self) -> str:
        """Tự sinh ID tiếp theo: ACC_01, ACC_02, ..."""
        existing = {acc.get("id", "") for acc in self._accounts}
        i = 1
        while True:
            candidate = f"ACC_{i:02d}"
            if candidate not in existing:
                return candidate
            i += 1

    def _add_account(self):
        new_id = self._next_id()
        dlg = AccountDialog(self, title="➕ Thêm tài khoản mới",
                            data={"id": new_id, "server": "S1"},
                            readonly_id=True)
        self.wait_window(dlg)
        if dlg.result:
            self._accounts.append(dlg.result)
            self._refresh_table()
            self._log(f"➕ Đã thêm: {dlg.result['username']} (ID: {dlg.result['id']})")

    def _edit_account(self):
        idx = self._selected_index()
        if idx is None:
            messagebox.showinfo("Chưa chọn", "Hãy chọn 1 tài khoản để sửa!")
            return
        acc = self._accounts[idx]
        dlg = AccountDialog(self, title=f"✏ Sửa: {acc['username']}", data=acc)
        self.wait_window(dlg)
        if dlg.result:
            self._accounts[idx] = dlg.result
            self._refresh_table()
            self._log(f"✏ Đã sửa: {dlg.result['username']}")

    def _delete_account(self):
        idx = self._selected_index()
        if idx is None:
            messagebox.showinfo("Chưa chọn", "Hãy chọn 1 tài khoản để xóa!")
            return
        acc = self._accounts[idx]
        if messagebox.askyesno("Xác nhận xóa",
                               f"Xóa tài khoản '{acc['username']}'?"):
            self._accounts.pop(idx)
            self._refresh_table()
            self._log(f"🗑 Đã xóa: {acc['username']}")


# ══════════════════════════════════════════════════════════════
# Tab 3: Tiến độ tài khoản (ProgressTab)
# ══════════════════════════════════════════════════════════════

class ProgressTab(tk.Frame):
    """Tab xem chi tiết tiến độ hoàn thành nhiệm vụ của từng tài khoản."""

    def __init__(self, parent, log_fn=None, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._log = log_fn
        self._auto_refresh = tk.BooleanVar(value=True)
        self._selected_acc_id: str | None = None
        self._data_cache: dict[str, dict] = {}
        self._build()
        self.after(200, self._refresh_data)
        self._schedule_auto_refresh()

    def _build(self):
        # 1. Summary Cards Frame
        summary_frame = tk.Frame(self, bg=BG, pady=6, padx=12)
        summary_frame.pack(fill="x")

        self._lbl_total_accs = tk.Label(summary_frame, text="👥 Tài khoản: 0", font=FONT_B, fg=WHITE, bg=ACCENT, padx=10, pady=4)
        self._lbl_total_accs.pack(side="left", padx=(0, 6))

        self._lbl_running_accs = tk.Label(summary_frame, text="🚀 Đang chạy: 0", font=FONT_B, fg="#38bdf8", bg=BG2, padx=10, pady=4)
        self._lbl_running_accs.pack(side="left", padx=(0, 6))

        self._lbl_finished_accs = tk.Label(summary_frame, text="🏆 Đã xong (20/20): 0", font=FONT_B, fg=GREEN, bg=BG2, padx=10, pady=4)
        self._lbl_finished_accs.pack(side="left", padx=(0, 6))

        # Controls (Auto-refresh + Refresh button)
        ctrl_frame = tk.Frame(summary_frame, bg=BG)
        ctrl_frame.pack(side="right")

        tk.Checkbutton(ctrl_frame, text="Tự động cập nhật (2s)", variable=self._auto_refresh,
                       font=FONT_SM, fg=WHITE, bg=BG, activebackground=BG,
                       selectcolor=ACCENT, cursor="hand2").pack(side="left", padx=6)

        tk.Button(ctrl_frame, text="🔄 Làm mới", font=FONT_SM, bg=ACCENT2, fg=WHITE,
                  activebackground=ACCENT, relief="flat", padx=10, pady=2, cursor="hand2",
                  command=self._refresh_data).pack(side="left")

        # 2. Bảng tổng quan tiến độ các tài khoản (Treeview)
        table_lbl = tk.Label(self, text="📋 Danh sách tiến độ tất cả tài khoản (click để xem chi tiết):", font=FONT_B, fg=WHITE, bg=BG, padx=12)
        table_lbl.pack(anchor="w", pady=(4, 0))

        table_frame = tk.Frame(self, bg=BG2, padx=8, pady=6)
        table_frame.pack(fill="x", padx=12, pady=4)

        cols = ("id", "username", "server", "progress", "percent", "status", "last_update")
        col_labels = ("ID", "Tài khoản", "Server", "Nhiệm vụ", "Tỷ lệ", "Trạng thái", "Cập nhật")
        col_widths = (65, 110, 60, 85, 65, 240, 75)

        self._tree_accs = ttk.Treeview(
            table_frame, columns=cols, show="headings",
            height=4, selectmode="browse"
        )
        for col, label, w in zip(cols, col_labels, col_widths):
            self._tree_accs.heading(col, text=label, anchor="w")
            self._tree_accs.column(col, width=w, minwidth=w, anchor="w")

        scroller_accs = ttk.Scrollbar(table_frame, orient="vertical", command=self._tree_accs.yview)
        self._tree_accs.configure(yscrollcommand=scroller_accs.set)
        self._tree_accs.pack(side="left", fill="x", expand=True)
        scroller_accs.pack(side="right", fill="y")
        self._tree_accs.bind("<<TreeviewSelect>>", self._on_select_account)

        # 3. Card chi tiết tài khoản đang chọn
        self._detail_frame = tk.LabelFrame(
            self, text=" 🎯 Chi tiết tài khoản đang chọn ",
            font=FONT_B, fg=WHITE, bg=BG2, padx=10, pady=8
        )
        self._detail_frame.pack(fill="both", expand=True, padx=12, pady=(6, 8))

        # Header info & progress bar
        top_detail = tk.Frame(self._detail_frame, bg=BG2)
        top_detail.pack(fill="x", pady=(0, 4))

        self._lbl_acc_title = tk.Label(top_detail, text="Chưa chọn tài khoản", font=FONT_H, fg=WHITE, bg=BG2)
        self._lbl_acc_title.pack(side="left")

        self._lbl_quest_stat = tk.Label(top_detail, text="0 / 20 (0%)", font=FONT_B, fg=YELLOW, bg=BG2)
        self._lbl_quest_stat.pack(side="right")

        # Progress bar
        prog_bar_frame = tk.Frame(self._detail_frame, bg=BG2)
        prog_bar_frame.pack(fill="x", pady=4)

        self._progress_bar = ttk.Progressbar(prog_bar_frame, orient="horizontal", mode="determinate", maximum=20)
        self._progress_bar.pack(fill="x", ipady=3)

        self._lbl_curr_activity = tk.Label(
            self._detail_frame, text="Nhiệm vụ hiện tại: Đang chờ khởi động...",
            font=FONT_SM, fg="#93c5fd", bg=BG2, anchor="w"
        )
        self._lbl_curr_activity.pack(fill="x", pady=(2, 6))

        # 4. Bảng lịch sử các nhiệm vụ đã hoàn thành của tài khoản
        tk.Label(self._detail_frame, text="📜 Lịch sử nhiệm vụ hoàn thành hôm nay:", font=FONT_SM, fg=GRAY, bg=BG2).pack(anchor="w")

        hist_frame = tk.Frame(self._detail_frame, bg=BG3)
        hist_frame.pack(fill="both", expand=True, pady=(2, 0))

        h_cols = ("num", "type", "detail", "status", "time")
        h_labels = ("NV #", "Loại nhiệm vụ", "Nội dung / Mục tiêu", "Kết quả", "Thời gian")
        h_widths = (55, 120, 310, 100, 75)

        self._tree_hist = ttk.Treeview(
            hist_frame, columns=h_cols, show="headings",
            height=5, selectmode="browse"
        )
        for col, label, w in zip(h_cols, h_labels, h_widths):
            self._tree_hist.heading(col, text=label, anchor="w")
            self._tree_hist.column(col, width=w, minwidth=w, anchor="w")

        scroller_hist = ttk.Scrollbar(hist_frame, orient="vertical", command=self._tree_hist.yview)
        self._tree_hist.configure(yscrollcommand=scroller_hist.set)
        self._tree_hist.pack(side="left", fill="both", expand=True)
        scroller_hist.pack(side="right", fill="y")

    def _schedule_auto_refresh(self):
        try:
            if self._auto_refresh.get():
                self._refresh_data()
        except Exception:
            pass
        self.after(2000, self._schedule_auto_refresh)

    def _refresh_data(self):
        import json
        accounts = []
        try:
            with open(ACCOUNTS_FILE, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                accounts = data.get("accounts", [])
        except Exception:
            pass

        total_done_all = 0
        running_cnt = 0
        finished_cnt = 0

        runtime_dir = Path("runtime")
        runtime_dir.mkdir(parents=True, exist_ok=True)

        rows_data = []
        for acc in accounts:
            acc_id = acc.get("id", "")
            username = acc.get("username", "")
            server = acc.get("server", "S1")

            prog_file = runtime_dir / f"progress_{acc_id}.json"
            acc_data = None
            if prog_file.exists():
                try:
                    with open(prog_file, "r", encoding="utf-8") as f:
                        acc_data = json.load(f)
                except Exception:
                    pass

            if not acc_data:
                # Kiểm tra state.json
                state_file = runtime_dir / "state.json"
                if state_file.exists():
                    try:
                        with open(state_file, "r", encoding="utf-8") as f:
                            s = json.load(f)
                            if s.get("account_id") == acc_id:
                                acc_data = {
                                    "account_id": acc_id,
                                    "username": username,
                                    "server": server,
                                    "total_quests_done": s.get("total_quests_done", 0),
                                    "status_text": s.get("current_state", "IDLE"),
                                    "current_quest_detail": s.get("extra", {}).get("quest_text", ""),
                                    "last_updated": s.get("timestamp", "")[-8:],
                                    "history": []
                                }
                    except Exception:
                        pass

            if not acc_data:
                acc_data = {
                    "account_id": acc_id,
                    "username": username,
                    "server": server,
                    "total_quests_done": 0,
                    "status_text": "Chưa chạy",
                    "current_quest_detail": "",
                    "last_updated": "--:--:--",
                    "history": []
                }

            self._data_cache[acc_id] = acc_data
            done = acc_data.get("total_quests_done", 0)
            total_done_all += done
            st = acc_data.get("status_text", "")
            if done >= 20:
                finished_cnt += 1
            elif "làm" in st.lower() or "chạy" in st.lower() or "đối thoại" in st.lower() or "đang" in st.lower():
                running_cnt += 1

            rows_data.append((
                acc_id,
                username,
                server,
                f"{done} / 20",
                f"{int(done / 20 * 100)}%",
                st,
                acc_data.get("last_updated", "--:--:--")
            ))

        # Cập nhật Summary
        self._lbl_total_accs.config(text=f"👥 Tài khoản: {len(accounts)}")
        self._lbl_running_accs.config(text=f"🚀 Đang chạy: {running_cnt}")
        self._lbl_finished_accs.config(text=f"🏆 Đã xong (20/20): {finished_cnt}")

        # Cập nhật Treeview danh sách tài khoản
        cur_sel = self._tree_accs.selection()
        sel_id = None
        if cur_sel:
            item_vals = self._tree_accs.item(cur_sel[0]).get("values", [])
            if item_vals:
                sel_id = str(item_vals[0])
        if not sel_id:
            sel_id = self._selected_acc_id

        self._tree_accs.delete(*self._tree_accs.get_children())
        sel_item = None
        for row in rows_data:
            item = self._tree_accs.insert("", "end", values=row)
            if sel_id and str(row[0]) == str(sel_id):
                sel_item = item

        if not sel_item and self._tree_accs.get_children():
            sel_item = self._tree_accs.get_children()[0]

        if sel_item:
            self._tree_accs.selection_set(sel_item)
            vals = self._tree_accs.item(sel_item).get("values", [])
            if vals:
                self._update_detail_view(str(vals[0]))

    def _on_select_account(self, event=None):
        sel = self._tree_accs.selection()
        if not sel:
            return
        vals = self._tree_accs.item(sel[0]).get("values", [])
        if vals:
            self._selected_acc_id = str(vals[0])
            self._update_detail_view(self._selected_acc_id)

    def _update_detail_view(self, acc_id: str):
        self._selected_acc_id = acc_id
        acc_data = self._data_cache.get(acc_id, {})
        if not acc_data:
            return

        username = acc_data.get("username", acc_id)
        server = acc_data.get("server", "S1")
        done = acc_data.get("total_quests_done", 0)
        pct = min(100, int(done / 20 * 100))

        color = GREEN if done >= 20 else (YELLOW if done > 0 else WHITE)
        self._lbl_acc_title.config(text=f"{acc_id} — {username} (Server: {server})")
        self._lbl_quest_stat.config(text=f"{done} / 20 nhiệm vụ ({pct}%)", fg=color)
        self._progress_bar["value"] = done

        curr_act = acc_data.get("status_text", "")
        detail_msg = acc_data.get("current_quest_detail", "")
        if detail_msg and detail_msg != curr_act:
            curr_act += f" — {detail_msg}"
        self._lbl_curr_activity.config(text=f"📌 Hiện tại: {curr_act}")

        # Lịch sử
        self._tree_hist.delete(*self._tree_hist.get_children())
        history = acc_data.get("history", [])
        if not history:
            self._tree_hist.insert("", "end", values=("-", "Chưa có", "Chưa có nhiệm vụ nào hoàn thành", "-", "--:--:--"))
        else:
            for h in history:
                self._tree_hist.insert("", "end", values=(
                    f"#{h.get('num', '-')}",
                    h.get("type", "NPC"),
                    h.get("detail", ""),
                    h.get("status", "Thành công ✅"),
                    h.get("time", "--:--:--")
                ))


# ══════════════════════════════════════════════════════════════
# Tab 4: Nhật ký (LogTab) — Lọc bỏ hoàn toàn DEBUG & WARNING
# ══════════════════════════════════════════════════════════════

class LogTab(tk.Frame):
    """
    Tab 4: Xem toàn bộ nhật ký hoạt động theo thời gian thực.
    Lọc bỏ triệt để các log DEBUG và WARNING, chỉ hiển thị:
    - Các sự kiện hoạt động nhiệm vụ (✅ Hoàn thành, 🏃 Di chuyển, 🗣️ Đối thoại, 🛍️ Mua đồ, ⚔️ Đánh quái,...)
    - Các bước chuyển trạng thái INFO quan trọng
    - Các thông báo lỗi ERROR để người dùng dễ theo dõi và xử lý
    """

    LOG_DIR = Path("logs")
    LOGURU_PATTERN = re.compile(
        r"^(?:\d{4}-\d{2}-\d{2}\s+)?(\d{2}:\d{2}:\d{2})(?:\.\d+)?\s*\|\s*([A-Z]+)\s*\|\s*(?:[^|\n]+?\s*-\s*)?(.*)$"
    )
    ACTIVITY_PATTERN = re.compile(
        r"^\[(\d{2}:\d{2}:\d{2})\]\s+(.*)$"
    )

    def __init__(self, parent, log_fn=None):
        super().__init__(parent, bg=BG)
        self._log_fn = log_fn
        self._auto_scroll = tk.BooleanVar(value=True)
        self._search_var = tk.StringVar(value="")
        self._date_var = tk.StringVar(value="")
        self._acc_var = tk.StringVar(value="Tất cả tài khoản")
        self._type_var = tk.StringVar(value="Tất cả (Bỏ Debug/Warning)")

        self._active_file: Path | None = None
        self._file_mtime: float = 0.0
        self._all_lines: list[str] = []
        self._is_polling = True

        self._build_ui()
        self._init_date_list()
        self._load_accounts_list()
        self._reload_file()
        self._schedule_poll()

    def _build_ui(self):
        # ── Hàng điều khiển 1: Ngày, Loại log, Làm mới, Thư mục log ──
        ctrl1 = tk.Frame(self, bg=BG, padx=12, pady=6)
        ctrl1.pack(fill="x")

        tk.Label(ctrl1, text="📅 Ngày:", font=FONT_SM, fg=GRAY, bg=BG).pack(side="left", padx=(0, 4))
        self._cb_date = ttk.Combobox(ctrl1, textvariable=self._date_var, width=12, font=FONT_SM, state="readonly")
        self._cb_date.pack(side="left", padx=(0, 10))
        self._cb_date.bind("<<ComboboxSelected>>", lambda e: self._reload_file())

        tk.Label(ctrl1, text="🎯 Loại log:", font=FONT_SM, fg=GRAY, bg=BG).pack(side="left", padx=(0, 4))
        self._cb_type = ttk.Combobox(
            ctrl1, textvariable=self._type_var, width=22, font=FONT_SM, state="readonly",
            values=[
                "Tất cả (Bỏ Debug/Warning)",
                "Chỉ Hoạt động & Nhiệm vụ",
                "Chỉ Thông báo Lỗi (Errors)",
            ]
        )
        self._cb_type.pack(side="left", padx=(0, 10))
        self._cb_type.bind("<<ComboboxSelected>>", lambda e: self._apply_filter())

        tk.Button(
            ctrl1, text="🔄 Làm mới", font=FONT_SM, bg=ACCENT, fg=WHITE,
            activebackground=ACCENT2, activeforeground=WHITE,
            relief="flat", padx=8, cursor="hand2", command=self._reload_file
        ).pack(side="left", padx=4)

        tk.Button(
            ctrl1, text="📁 Thư mục log", font=FONT_SM, bg=GRAY2, fg=WHITE,
            activebackground=GRAY, activeforeground=WHITE,
            relief="flat", padx=8, cursor="hand2", command=self._open_log_folder
        ).pack(side="right")

        # ── Hàng điều khiển 2: Bộ lọc theo từng Tài khoản (Combobox chọn tài khoản) ──
        acc_bar = tk.Frame(self, bg=BG2, padx=12, pady=5)
        acc_bar.pack(fill="x", padx=12, pady=(2, 4))

        tk.Label(acc_bar, text="👤 Lọc tài khoản:", font=FONT_B, fg=YELLOW, bg=BG2).pack(side="left", padx=(0, 8))

        # Combobox chọn tài khoản (hỗ trợ chọn bất kỳ tài khoản nào trong toàn bộ danh sách)
        self._cb_acc = ttk.Combobox(acc_bar, textvariable=self._acc_var, width=28, font=FONT_SM, state="readonly")
        self._cb_acc.pack(side="left", padx=(0, 8))
        self._cb_acc.bind("<<ComboboxSelected>>", lambda e: self._on_acc_combobox_selected())

        # Nút xem tất cả để reset nhanh
        self._btn_all_acc = tk.Button(
            acc_bar, text="🌐 Xem tất cả",
            font=FONT_SM, bg=ACCENT, fg=WHITE,
            activebackground=ACCENT2, activeforeground=WHITE,
            relief="flat", padx=10, pady=2, cursor="hand2",
            command=lambda: self._select_account("Tất cả tài khoản")
        )
        self._btn_all_acc.pack(side="left", padx=(0, 10))

        # Nhãn hiển thị tổng số tài khoản
        self._lbl_acc_total = tk.Label(acc_bar, text="", font=FONT_SM, fg=GRAY, bg=BG2)
        self._lbl_acc_total.pack(side="left", padx=4)

        # ── Hàng điều khiển 3: Tìm kiếm, Auto-scroll, Thống kê, Xoá ──
        ctrl2 = tk.Frame(self, bg=BG, padx=12, pady=4)
        ctrl2.pack(fill="x")

        tk.Label(ctrl2, text="🔍 Tìm:", font=FONT_SM, fg=GRAY, bg=BG).pack(side="left", padx=(0, 4))
        ent_search = tk.Entry(ctrl2, textvariable=self._search_var, font=FONT_SM,
                              bg=BG2, fg=WHITE, insertbackground=WHITE, relief="flat", width=22)
        ent_search.pack(side="left", padx=(0, 4))
        ent_search.bind("<KeyRelease>", lambda e: self._apply_filter())

        tk.Button(
            ctrl2, text="✗", font=FONT_SM, bg=GRAY2, fg=WHITE, relief="flat",
            padx=5, cursor="hand2", command=self._clear_search
        ).pack(side="left", padx=(0, 12))

        tk.Checkbutton(
            ctrl2, text="⬇️ Tự động cuộn", variable=self._auto_scroll,
            font=FONT_SM, fg=WHITE, bg=BG, activebackground=BG,
            selectcolor=ACCENT
        ).pack(side="left", padx=(0, 10))

        self._lbl_count = tk.Label(ctrl2, text="Hiển thị: 0 dòng", font=FONT_SM, fg=GRAY, bg=BG)
        self._lbl_count.pack(side="left", padx=10)

        tk.Button(
            ctrl2, text="🧹 Xoá xem", font=FONT_SM, bg=GRAY2, fg=WHITE,
            relief="flat", padx=8, cursor="hand2", command=self._clear_view
        ).pack(side="right")

        # ── Vùng hiển thị Text Log ──
        view_frame = tk.Frame(self, bg=BG3, padx=2, pady=2)
        view_frame.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        scroll_y = ttk.Scrollbar(view_frame, orient="vertical")
        scroll_y.pack(side="right", fill="y")

        self._txt = tk.Text(
            view_frame, bg=BG3, fg=WHITE, font=FONT_C,
            wrap="word", relief="flat", state="disabled",
            yscrollcommand=scroll_y.set
        )
        self._txt.pack(side="left", fill="both", expand=True)
        scroll_y.config(command=self._txt.yview)

        # Cấu hình màu sắc syntax cho text
        self._txt.tag_configure("tag_time", foreground="#64748b")
        self._txt.tag_configure("tag_success", foreground="#4ade80", font=("Consolas", 9, "bold"))
        self._txt.tag_configure("tag_error", foreground="#f87171", font=("Consolas", 9, "bold"))
        self._txt.tag_configure("tag_activity", foreground="#38bdf8")
        self._txt.tag_configure("tag_account", foreground="#fbbf24", font=("Consolas", 9, "bold"))
        self._txt.tag_configure("tag_info", foreground="#cbd5e1")
        self._txt.tag_configure("tag_header", foreground="#a78bfa", font=("Consolas", 9, "bold"))
        self._txt.tag_configure("tag_dim", foreground="#475569")

    def _init_date_list(self):
        """Quét các file .log trong thư mục logs/ và chọn file mới nhất."""
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_files = sorted(self.LOG_DIR.glob("*.log"), reverse=True)
        dates = [f.stem for f in log_files]
        if not dates:
            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")
            dates = [today]
        self._cb_date["values"] = dates
        if dates:
            self._date_var.set(dates[0])

    def _select_account(self, acc_name: str):
        """Chọn tài khoản cần lọc qua combobox hoặc nút Xem tất cả."""
        self._acc_var.set(acc_name)
        if hasattr(self, "_btn_all_acc"):
            if acc_name in ("Tất cả", "Tất cả tài khoản"):
                self._btn_all_acc.config(bg=ACCENT, fg=WHITE)
            else:
                self._btn_all_acc.config(bg=BG3, fg=GRAY)
        self._apply_filter()

    def _on_acc_combobox_selected(self):
        val = self._acc_var.get()
        self._select_account(val)

    def _load_accounts_list(self):
        """Nạp danh sách tất cả tài khoản vào combobox."""
        acc_opts = ["Tất cả tài khoản"]
        count = 0
        if ACCOUNTS_FILE.exists():
            try:
                with open(ACCOUNTS_FILE, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                for a in data.get("accounts", []):
                    acc_id = a.get("id", "")
                    uname = a.get("username", "")
                    if acc_id:
                        lbl = f"{acc_id} ({uname})" if uname else acc_id
                        acc_opts.append(lbl)
                        count += 1
            except Exception:
                pass

        self._cb_acc["values"] = acc_opts
        if hasattr(self, "_lbl_acc_total"):
            self._lbl_acc_total.config(text=f"(Tổng {count} tài khoản)")

        cur_acc = self._acc_var.get()
        if not cur_acc or cur_acc not in acc_opts:
            cur_acc = "Tất cả tài khoản"
            self._acc_var.set(cur_acc)
        self._select_account(cur_acc)

    def _get_selected_log_path(self) -> Path | None:
        dt = self._date_var.get().strip()
        if not dt:
            return None
        return self.LOG_DIR / f"{dt}.log"

    def _reload_file(self):
        """Đọc toàn bộ file log hiện tại."""
        p = self._get_selected_log_path()
        self._active_file = p
        self._all_lines = []
        if p and p.exists():
            try:
                self._file_mtime = p.stat().st_mtime
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    self._all_lines = [l.rstrip("\r\n") for l in f]
            except Exception as e:
                self._all_lines = [f"[Lỗi đọc file log: {e}]"]
        else:
            self._all_lines = ["[Chưa có dữ liệu log cho ngày này]"]

        self._apply_filter()

    def _is_allowed(self, line: str) -> bool:
        """Kiểm tra điều kiện hiển thị: BỎ HOÀN TOÀN DEBUG VÀ WARNING."""
        # 1. Loại bỏ triệt để DEBUG và WARNING
        if (
            "| DEBUG " in line or "| WARNING " in line
            or "| DEBUG|" in line or "| WARNING|" in line
            or "DEBUG    |" in line or "WARNING  |" in line
            or " - DEBUG - " in line or " - WARNING - " in line
        ):
            return False

        # 2. Lọc theo Loại log
        ltype = self._type_var.get()
        if ltype == "Chỉ Hoạt động & Nhiệm vụ":
            # Chỉ nhận các dòng có icon hoạt động hoặc thông tin nhiệm vụ
            has_activity = any(ic in line for ic in (
                "✅", "❌", "🏃", "🗣️", "🛍️", "⚔️", "🏆", "▶️", "⏹️", "💬", "🏁", "⚠️",
                "HOÀN THÀNH", "Nhiệm vụ", "nhan nhiem vu", "doi thoai", "da hoan thanh"
            ))
            if not has_activity:
                return False
        elif ltype == "Chỉ Thông báo Lỗi (Errors)":
            has_err = any(er in line for er in (
                "ERROR", "BotError", "Exception", "LỖI", "Traceback", "failed", "device not found", "cannot"
            ))
            if not has_err:
                return False

        # 3. Lọc theo Tài khoản
        facc = self._acc_var.get()
        if facc and facc not in ("Tất cả", "Tất cả tài khoản"):
            acc_token = facc.split()[0].lower().strip("()") # e.g. acc_01
            uname_token = ""
            if "(" in facc and ")" in facc:
                uname_token = facc.split("(")[1].split(")")[0].strip().lower()

            line_lower = line.lower()
            match_acc = (
                (acc_token and (f"[{acc_token}]" in line_lower or acc_token in line_lower))
                or (uname_token and (f"[{uname_token}]" in line_lower or uname_token in line_lower))
            )
            if not match_acc:
                return False

        # 4. Lọc theo ô Tìm kiếm
        kw = self._search_var.get().strip().lower()
        if kw and kw not in line.lower():
            return False

        return True

    def _apply_filter(self):
        """Lọc và vẽ lại text log lên màn hình."""
        self._txt.config(state="normal")
        self._txt.delete("1.0", "end")

        filtered = [l for l in self._all_lines if self._is_allowed(l)]
        # Giới hạn 1500 dòng mới nhất để giao diện luôn mượt mà
        if len(filtered) > 1500:
            filtered = filtered[-1500:]

        for line in filtered:
            self._render_line(line)

        facc = self._acc_var.get()
        if facc and facc not in ("Tất cả", "Tất cả tài khoản"):
            self._lbl_count.config(text=f"Hiển thị: {len(filtered)} dòng ({facc})")
        else:
            self._lbl_count.config(text=f"Hiển thị: {len(filtered)} dòng")

        if self._auto_scroll.get():
            self._txt.see("end")

        self._txt.config(state="disabled")

        if self._auto_scroll.get():
            self._txt.see("end")

        self._txt.config(state="disabled")

    def _render_line(self, line: str):
        """
        Ẩn bớt cột ngày và cột 'src...' để chỉ hiển thị thời gian và nội dung dễ đọc.
        VD: '2026-09-14 22:37:28.675 | INFO | src.bot_controller:... - [ACC_02] ...'
        -> '[22:37:28] [ACC_02] ...'
        """
        m_log = self.LOGURU_PATTERN.match(line)
        m_act = self.ACTIVITY_PATTERN.match(line)

        time_str = ""
        content = line
        level = ""

        if m_log:
            time_str = m_log.group(1)
            level = m_log.group(2).strip()
            content = m_log.group(3).strip()
            if level == "ERROR":
                content = f"❌ [LỖI] {content}"
            elif level == "WARNING":
                content = f"⚠️ {content}"
        elif m_act:
            time_str = m_act.group(1)
            content = m_act.group(2).strip()

        tag = "tag_info"
        if any(e in content for e in ("ERROR", "❌", "Traceback", "BotError", "LỖI:", "[LỖI]")):
            tag = "tag_error"
        elif any(s in content for s in ("✅", "HOÀN THÀNH", "thành công", "🏆")):
            tag = "tag_success"
        elif any(a in content for a in ("🏃", "🗣️", "🛍️", "⚔️", "💬", "▶️", "⏹️", "📱", "🎮", "👤", "⏳", "🏁", "📋")):
            tag = "tag_activity"
        elif content.startswith("===") or "Tool khởi động" in content:
            tag = "tag_header"

        if time_str:
            self._txt.insert("end", f"[{time_str}] ", "tag_time")
            # Tô màu vàng nổi bật cho tag tài khoản nếu có, vd: [ACC_01] hoặc [hk0001]
            m_acc_prefix = re.match(r"^(\[[A-Za-z0-9_]+\])\s*(.*)$", content)
            if m_acc_prefix:
                self._txt.insert("end", f"{m_acc_prefix.group(1)} ", "tag_account")
                self._txt.insert("end", m_acc_prefix.group(2) + "\n", tag)
            else:
                self._txt.insert("end", content + "\n", tag)
        else:
            self._txt.insert("end", content + "\n", tag)

    def _schedule_poll(self):
        """Kiểm tra file log mỗi 1.5s để cập nhật dòng mới."""
        if not self._is_polling:
            return
        try:
            p = self._active_file
            if p and p.exists():
                cur_mtime = p.stat().st_mtime
                if cur_mtime > self._file_mtime:
                    self._file_mtime = cur_mtime
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        lines = [l.rstrip("\r\n") for l in f]
                    
                    if len(lines) > len(self._all_lines):
                        new_lines = lines[len(self._all_lines):]
                        self._all_lines = lines

                        # Thêm các dòng mới hợp lệ vào cuối
                        added = 0
                        self._txt.config(state="normal")
                        for nl in new_lines:
                            if self._is_allowed(nl):
                                self._render_line(nl)
                                added += 1
                        
                        if added > 0 and self._auto_scroll.get():
                            self._txt.see("end")
                        self._txt.config(state="disabled")
        except Exception:
            pass

        self.after(1500, self._schedule_poll)

    def _clear_search(self):
        self._search_var.set("")
        self._apply_filter()

    def _clear_view(self):
        self._txt.config(state="normal")
        self._txt.delete("1.0", "end")
        self._txt.config(state="disabled")
        self._lbl_count.config(text="Hiển thị: 0 dòng")

    def _open_log_folder(self):
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(self.LOG_DIR.resolve()))
        except Exception:
            messagebox.showinfo("Thư mục Log", f"Đường dẫn: {self.LOG_DIR.resolve()}", parent=self)


# ══════════════════════════════════════════════════════════════
# Main App
# ══════════════════════════════════════════════════════════════

class LauncherApp(tk.Tk):
    def __init__(self):
        super().__init__()

        loader = ConfigLoader()
        loader.load_all()
        self._cfg = loader.load_app()
        self._ld = LDPlayerController(self._cfg)

        self.title("AutoQuestTool — Launcher")
        self.configure(bg=BG)
        self.resizable(False, False)
        self._center_window(820, 750)
        self._build()

    def _center_window(self, w: int, h: int):
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=ACCENT, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🎮  AutoQuestTool — Launcher",
                 font=FONT_H, fg=WHITE, bg=ACCENT).pack()
        tk.Label(hdr, text="Tình Kiếm Truyền Kỳ · LDPlayer 9",
                 font=FONT_SM, fg=GRAY, bg=ACCENT).pack()

        # Log box TRƯỚC tabs để _log() hoạt động ngay từ đầu
        log_frame = tk.Frame(self, bg=BG, padx=12, pady=4)
        log_frame.pack(fill="x", side="bottom")
        tk.Label(log_frame, text="Trạng thái:", font=FONT_SM, fg=GRAY, bg=BG).pack(anchor="w")
        self._log_box = tk.Text(
            log_frame, height=3, bg=BG3, fg=GREEN,
            font=FONT_C, relief="flat", state="disabled", wrap="word"
        )
        self._log_box.pack(fill="x")

        # Tabs
        tab_style = ttk.Style()
        tab_style.configure("TNotebook", background=BG, borderwidth=0)
        tab_style.configure("TNotebook.Tab",
            background=GRAY2, foreground=WHITE,
            font=FONT_B, padding=(16, 6))
        tab_style.map("TNotebook.Tab",
            background=[("selected", ACCENT)],
            foreground=[("selected", WHITE)])

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=0, pady=0)

        # Tab 1: Máy ảo
        tab1 = EmulatorTab(notebook, self._ld, self._log, app_cfg=self._cfg)
        notebook.add(tab1, text="  📱 Máy ảo  ")

        # Tab 2: Tài khoản
        tab2 = AccountsTab(notebook, self._log)
        notebook.add(tab2, text="  👤 Tài khoản  ")

        # Tab 3: Tiến độ tài khoản
        tab3 = ProgressTab(notebook, self._log)
        notebook.add(tab3, text="  📊 Tiến độ  ")

        # Tab 4: Nhật ký (lọc bỏ DEBUG & WARNING)
        tab4 = LogTab(notebook, self._log)
        notebook.add(tab4, text="  📜 Nhật ký  ")

    def _log(self, msg: str):
        self._log_box.config(state="normal")
        self._log_box.insert("end", msg + "\n")
        self._log_box.see("end")
        self._log_box.config(state="disabled")


def main():
    app = LauncherApp()
    app.mainloop()


if __name__ == "__main__":
    main()
