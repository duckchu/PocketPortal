"""Tests for the i18n helper and the persona loader."""

import json
from pathlib import Path

import pytest

from core import i18n
from illust import comfy_gen

ROOT = Path(__file__).resolve().parent.parent


def test_locale_files_exist():
    assert (ROOT / "i18n" / "en.json").is_file()
    assert (ROOT / "i18n" / "zh-TW.json").is_file()


def test_en_locale_is_identity():
    en = json.loads((ROOT / "i18n" / "en.json").read_text(encoding="utf-8"))
    assert en, "en.json must not be empty"
    for key, value in en.items():
        assert key == value, f"en.json must map a string to itself: {key!r}"


def test_zh_locale_keys_are_english():
    zh = json.loads((ROOT / "i18n" / "zh-TW.json").read_text(encoding="utf-8"))
    assert zh, "zh-TW.json must not be empty"
    for key in zh:
        assert not any("\u4e00" <= ch <= "\u9fff" for ch in key), f"non-English key: {key!r}"


def test_translate_roundtrip():
    i18n.set_locale("en")
    assert i18n.t("⏳ Generating reply...") == "⏳ Generating reply..."
    i18n.set_locale("zh-TW")
    zh_expected = json.loads((ROOT / "i18n" / "zh-TW.json").read_text(encoding="utf-8"))["⏳ Generating reply..."]
    assert i18n.t("⏳ Generating reply...") == zh_expected
    assert zh_expected != "⏳ Generating reply..."
    i18n.set_locale("en")


def test_unknown_key_falls_back_to_source():
    i18n.set_locale("zh-TW")
    assert i18n.t("Some string that is not translated") == "Some string that is not translated"
    i18n.set_locale("en")


def test_personas_absent_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("ST_PERSONAS_FILE", str(tmp_path / "missing.yaml"))
    assert comfy_gen.load_personas() == ({}, {})
    assert comfy_gen.resolve_persona_for_character("Anyone") is None


def test_personas_loaded_from_yaml(tmp_path, monkeypatch):
    yaml_file = tmp_path / "personas.yaml"
    yaml_file.write_text(
        "personas:\n"
        "  hero:\n"
        "    keywords: [\"Aria\"]\n"
        "    pos: [\"hero_pos.txt\"]\n"
        "    neg: []\n"
        "    lora: []\n"
        "name_map:\n"
        "  testhero: hero\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ST_PERSONAS_FILE", str(yaml_file))
    personas, name_map = comfy_gen.load_personas()
    assert "hero" in personas
    assert name_map == {"testhero": "hero"}
    assert comfy_gen.resolve_persona_for_character("Aria the scholar") == "hero"
    assert comfy_gen.resolve_persona_for_character("TestHero") == "hero"


def test_shipped_sources_contain_no_personal_personas():
    """The public sources must not ship third-party character definitions."""
    forbidden = json.loads((ROOT / "tests" / "forbidden_names.json").read_text(encoding="utf-8"))
    for rel in ("illust/comfy_gen.py", "illust/prompts", "config"):
        path = ROOT / rel
        files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
        for f in files:
            if f.suffix not in {".py", ".txt", ".yaml", ".json"}:
                continue
            text = f.read_text(encoding="utf-8", errors="ignore").lower()
            for name in forbidden:
                assert name not in text, f"personal persona reference leaked in {f}: {name}"
