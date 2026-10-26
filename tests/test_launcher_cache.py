from launcher import clear_runtime_checkpoint


def test_clear_runtime_checkpoint_deletes_only_state_file(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    state_file = runtime / "state.json"
    progress_file = runtime / "progress_ACC_03.json"
    state_file.write_text('{"current_state": "NAVIGATE"}', encoding="utf-8")
    progress_file.write_text('{"total_quests_done": 3}', encoding="utf-8")

    assert clear_runtime_checkpoint(state_file) is True
    assert not state_file.exists()
    assert progress_file.exists()


def test_clear_runtime_checkpoint_is_safe_when_already_empty(tmp_path):
    state_file = tmp_path / "runtime" / "state.json"

    assert clear_runtime_checkpoint(state_file) is False


def test_reset_daily_progress_resets_counts_and_deletes_state(tmp_path):
    from launcher import reset_daily_progress
    import json
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    state_file = runtime / "state.json"
    p1 = runtime / "progress_ACC_01.json"
    p2 = runtime / "progress_ACC_02.json"
    state_file.write_text('{"current_state": "NAVIGATE"}', encoding="utf-8")
    p1.write_text(json.dumps({"total_quests_done": 20, "status_text": "Đã xong 20/20"}), encoding="utf-8")
    p2.write_text(json.dumps({"total_quests_done": 10, "status_text": "Đang làm"}), encoding="utf-8")

    cnt = reset_daily_progress(runtime)
    assert cnt == 2
    assert not state_file.exists()
    d1 = json.loads(p1.read_text(encoding="utf-8"))
    d2 = json.loads(p2.read_text(encoding="utf-8"))
    assert d1["total_quests_done"] == 0
    assert d2["total_quests_done"] == 0
    assert "Ngày mới" in d1["status_text"]


