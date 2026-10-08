from pathlib import Path
import json

import pandas as pd

from aria.region import GPGL_REGION, SGP_REGION
from aria import storage
from aria.history import historical_plan


def _redirect(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "CONFIG_PATH", tmp_path / "storage.json")
    monkeypatch.setattr(storage, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(storage, "DEFAULT_ROOT", tmp_path / "data")


def test_v020_config_migrates_to_named_profile(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    legacy = {
        "version": 2,
        "enabled": True,
        "root": str(tmp_path / "data"),
        "region": GPGL_REGION.as_dict(),
        "sources": {"surface": {"enabled": True, "interval_minutes": 5}},
    }
    storage.CONFIG_PATH.write_text(json.dumps(legacy))
    cfg = storage.load_config()
    assert cfg["version"] == 3
    assert cfg["profile_name"] == "gpgl"
    assert "gpgl" in cfg["profiles"]
    assert json.loads(storage.CONFIG_PATH.read_text())["version"] == 3


def test_collector_runs_two_enabled_profiles(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    cfg = storage.default_config(GPGL_REGION)
    cfg["root"] = str(tmp_path / "data")
    cfg["enabled"] = True
    storage.save_config(cfg)
    storage.create_storage_profile("ARM SGP Live", SGP_REGION, copy_from="gpgl", enabled=True)
    gpgl = storage.load_config("gpgl")
    gpgl["enabled"] = True
    storage.save_config(gpgl)

    calls = []
    def fake_collect(source, region, directory):
        calls.append((source, region.name, str(directory)))
        return {"data_time": "2026-09-22T12:00:00Z", "files": []}
    monkeypatch.setattr(storage, "_collect", fake_collect)
    storage.update_storage(source="surface", force=True)

    assert {item[1] for item in calls} == {"GPGL", "ARM SGP"}
    assert any("/gpgl/" in item[2] for item in calls)
    assert any("/arm_sgp_live/" in item[2] for item in calls)


def test_history_plan_counts_steps(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    storage.save_config(storage.default_config(GPGL_REGION))
    plan = historical_plan(
        "gpgl", "2026-09-20T00:00Z", "2026-09-20T01:00Z", ("mrms", "hrrr")
    )
    assert plan["total_steps"] == 15
    assert {row["source"] for row in plan["sources"]} == {"mrms", "hrrr"}
