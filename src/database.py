"""
AutoQuestTool - Database Layer
SQLite wrapper cho lịch sử và trạng thái lâu dài.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from loguru import logger


DB_FILE = Path("runtime/autoquest.db")


class Database:
    """
    SQLite wrapper.
    Tạo schema tự động khi khởi tạo.
    """

    def __init__(self, db_file: Path = DB_FILE):
        self.db_file = db_file
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ============================
    # Connection
    # ============================

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_file))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Tạo tất cả bảng nếu chưa có."""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS accounts (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    server TEXT DEFAULT '',
                    status TEXT DEFAULT 'pending',
                    enabled INTEGER DEFAULT 1,
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS account_status (
                    account_id TEXT PRIMARY KEY,
                    current_state TEXT DEFAULT 'IDLE',
                    current_map TEXT DEFAULT '',
                    current_quest TEXT DEFAULT '',
                    total_quests INTEGER DEFAULT 0,
                    last_run TEXT,
                    notes TEXT DEFAULT '',
                    FOREIGN KEY (account_id) REFERENCES accounts(id)
                );

                CREATE TABLE IF NOT EXISTS quest_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    quest_type TEXT NOT NULL,
                    result TEXT NOT NULL,
                    map TEXT DEFAULT '',
                    npc TEXT DEFAULT '',
                    started_at TEXT,
                    completed_at TEXT,
                    notes TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS gold_transfer_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_account TEXT NOT NULL,
                    to_account TEXT NOT NULL,
                    amount INTEGER NOT NULL,
                    result TEXT NOT NULL,
                    timestamp TEXT
                );

                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT,
                    error_code TEXT,
                    state TEXT,
                    action TEXT,
                    message TEXT,
                    screenshot_path TEXT DEFAULT '',
                    retry_count INTEGER DEFAULT 0,
                    timestamp TEXT
                );

                CREATE TABLE IF NOT EXISTS run_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    started_at TEXT,
                    ended_at TEXT,
                    final_state TEXT,
                    quests_done INTEGER DEFAULT 0,
                    errors_count INTEGER DEFAULT 0
                );
            """)
        logger.debug("Database schema initialized")

    # ============================
    # Account Operations
    # ============================

    def upsert_account(
        self,
        account_id: str,
        username: str,
        server: str = "",
        status: str = "pending",
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO accounts (id, username, server, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    username=excluded.username,
                    server=excluded.server,
                    updated_at=excluded.updated_at
            """, (account_id, username, server, status, now, now))

    def update_account_status(
        self,
        account_id: str,
        current_state: str,
        current_map: str = "",
        current_quest: str = "",
        total_quests: Optional[int] = None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            # Upsert
            existing = conn.execute(
                "SELECT total_quests FROM account_status WHERE account_id=?",
                (account_id,)
            ).fetchone()
            if existing:
                q = total_quests if total_quests is not None else existing["total_quests"]
                conn.execute("""
                    UPDATE account_status SET
                        current_state=?, current_map=?, current_quest=?,
                        total_quests=?, last_run=?
                    WHERE account_id=?
                """, (current_state, current_map, current_quest, q, now, account_id))
            else:
                q = total_quests or 0
                conn.execute("""
                    INSERT INTO account_status
                        (account_id, current_state, current_map, current_quest, total_quests, last_run)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (account_id, current_state, current_map, current_quest, q, now))

    # ============================
    # Quest History
    # ============================

    def log_quest(
        self,
        account_id: str,
        quest_type: str,
        result: str,
        map_name: str = "",
        npc: str = "",
        started_at: str = "",
        notes: str = "",
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO quest_history
                    (account_id, quest_type, result, map, npc, started_at, completed_at, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (account_id, quest_type, result, map_name, npc, started_at, now, notes))

    # ============================
    # Gold Transfer
    # ============================

    def log_transfer(
        self,
        from_account: str,
        to_account: str,
        amount: int,
        result: str,
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO gold_transfer_history
                    (from_account, to_account, amount, result, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (from_account, to_account, amount, result, now))

    # ============================
    # Error Logging
    # ============================

    def log_error(
        self,
        account_id: str,
        error_code: str,
        state: str,
        action: str,
        message: str,
        screenshot_path: str = "",
        retry_count: int = 0,
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO errors
                    (account_id, error_code, state, action, message,
                     screenshot_path, retry_count, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (account_id, error_code, state, action, message,
                  screenshot_path, retry_count, now))

    # ============================
    # Run History
    # ============================

    def start_run(self, account_id: str) -> int:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            cursor = conn.execute("""
                INSERT INTO run_history (account_id, started_at, final_state)
                VALUES (?, ?, 'RUNNING')
            """, (account_id, now))
            return cursor.lastrowid

    def end_run(
        self,
        run_id: int,
        final_state: str,
        quests_done: int = 0,
        errors_count: int = 0,
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute("""
                UPDATE run_history SET
                    ended_at=?, final_state=?, quests_done=?, errors_count=?
                WHERE id=?
            """, (now, final_state, quests_done, errors_count, run_id))

    # ============================
    # Queries
    # ============================

    def get_account_summary(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT a.id, a.username, a.server,
                       COALESCE(s.current_state, 'pending') as state,
                       COALESCE(s.total_quests, 0) as total_quests,
                       s.last_run
                FROM accounts a
                LEFT JOIN account_status s ON a.id = s.account_id
                WHERE a.enabled = 1
                ORDER BY a.id
            """).fetchall()
            return [dict(r) for r in rows]
