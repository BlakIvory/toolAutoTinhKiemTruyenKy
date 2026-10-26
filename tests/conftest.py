# AutoQuestTool - conftest.py
import sys
from pathlib import Path
import pytest

# Thêm root vào path để import src.*
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(autouse=True)
def isolate_test_activity_logs(monkeypatch):
    """Ngăn các unit test ghi rác log giả lập vào file logs/YYYY-MM-DD.log của người dùng."""
    from src.activity_log import ActivityLog
    monkeypatch.setattr(ActivityLog, "_write", lambda self, icon, msg: None)
