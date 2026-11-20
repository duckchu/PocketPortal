"""Self-contained unit tests for PocketPortal core modules."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from core.context_engine import ContextEngine
from core.handlers import BotController
from core.llm_adapter import LLMAdapter
from core.storage import CharacterCard, LorebookEntry, StorageManager, Turn


def _make_stack(tmp_workspace: Path):
    for sub in ("config", "characters", "lorebooks", "sessions", "exports", "logs"):
        (tmp_workspace / sub).mkdir(parents=True, exist_ok=True)

    bot_cfg = {
        "bot_token": "test-token",
        "allowed_user_ids": [12345],
        "default_user_name": "Player",
        "koboldcpp_url": "http://127.0.0.1:5001",
        "koboldcpp_bin": str(tmp_workspace / "koboldcpp"),
        "ollama_url": "http://127.0.0.1:11434",
        "mlx_url": "http://127.0.0.1:8080",
        "active_preset": "default",
        "idle_unload_seconds": 300,
    }
    models_cfg = {
        "default_model": "local-kobold",
        "models": {
            "local-kobold": {
                "backend": "koboldcpp",
                "display_name": "[Kobold] Test 8B",
                "path": str(tmp_workspace / "test.gguf"),
                "vram_gb": 5.0,
                "context_size": 65536,
                "instruct_template": "chatml_think",
                "sampling": {"temp": 0.9, "max_length": 1024},
            },
            "local-ollama": {
                "backend": "ollama",
                "ollama_model": "test:8b",
                "display_name": "[Ollama] Test 8B",
                "vram_gb": 5.0,
                "context_size": 65536,
                "instruct_template": "chatml",
                "sampling": {"temp": 0.9, "max_length": 1024},
            },
            "local-mlx": {
                "backend": "mlx",
                "mlx_model": "mlx-community/test-4bit",
                "display_name": "[MLX] Test 8B (4bit)",
                "vram_gb": 5.0,
                "context_size": 65536,
                "instruct_template": "chatml",
                "sampling": {"temp": 0.9, "max_length": 1024},
            },
            "cloud-flash": {
                "backend": "ollama_cloud",
                "ollama_model": "deepseek-v4.1-flash:cloud",
                "display_name": "[Cloud] DeepSeek v4.1 Flash",
                "vram_gb": 0,
                "context_size": 65536,
                "instruct_template": "chatml_think",
                "sampling": {"temp": 0.9, "max_length": 1024},
            },
        },
    }
    presets_cfg = {
        "instruct_templates": {
            "chatml": {
                "bos_token": "",
                "system_prefix": "<|im_start|>system\n",
                "system_suffix": "<|im_end|>\n",
                "user_prefix": "<|im_start|>user\n",
                "user_suffix": "<|im_end|>\n",
                "assistant_prefix": "<|im_start|>assistant\n",
                "assistant_suffix": "<|im_end|>\n",
                "assistant_prefill": "<|im_start|>assistant\n",
                "stop_sequences": ["<|im_end|>"],
            },
            "chatml_think": {
                "bos_token": "",
                "system_prefix": "<|im_start|>system\n",
                "system_suffix": "<|im_end|>\n",
                "user_prefix": "<|im_start|>user\n",
                "user_suffix": "<|im_end|>\n",
                "assistant_prefix": "<|im_start|>assistant\n",
                "assistant_suffix": "<|im_end|>\n",
                "assistant_prefill": "<|im_start|>assistant\n<think>\n\n</think>\n",
                "stop_sequences": ["<|im_end|>"],
            },
        },
        "presets": {
            "default": {
                "system_prompt": "You are {{char}}, roleplaying with {{user}}.",
                "post_history_instructions": "Stay in character.",
                "terminal_instruction": "",
                "jailbreak_prefill": "",
            }
        },
        "regex_post_process": [],
    }
    (tmp_workspace / "config" / "bot_config.yaml").write_text(yaml.safe_dump(bot_cfg), encoding="utf-8")
    (tmp_workspace / "config" / "models.yaml").write_text(yaml.safe_dump(models_cfg), encoding="utf-8")
    (tmp_workspace / "config" / "system_presets.yaml").write_text(yaml.safe_dump(presets_cfg), encoding="utf-8")

    storage = StorageManager(tmp_workspace)
    kobold = LLMAdapter(bot_config=bot_cfg, models_config=models_cfg, logs_dir=storage.logs_dir)
    engine = ContextEngine(storage=storage, kobold=kobold, bot_config=bot_cfg, presets_config=presets_cfg)
    return storage, kobold, engine, bot_cfg


def test_storage_and_cli_args(tmp_path: Path) -> None:
    storage, kobold, _, _ = _make_stack(tmp_path)
    (tmp_path / "koboldcpp").write_bytes(b"bin")
    (tmp_path / "test.gguf").write_bytes(b"gguf")
    args = kobold.build_cli_args("local-kobold")
    assert "--contextsize" in args and "65536" in args
    mlx_args = kobold.build_mlx_cli_args("local-mlx")
    assert "mlx_lm" in mlx_args and "server" in mlx_args


def test_inline_ooc_parsing(tmp_path: Path) -> None:
    _, _, engine, _ = _make_stack(tmp_path)
    dlg, ooc = engine.parse_user_input("Hello there! (OOC: make her smile)")
    assert dlg == "Hello there!"
    assert ooc == "make her smile"

    dlg2, ooc2 = engine.parse_user_input("// speed up the scene")
    assert dlg2 == ""
    assert ooc2 == "speed up the scene"


@pytest.mark.asyncio
async def test_prompt_stack_and_story_arc(tmp_path: Path) -> None:
    storage, kobold, engine, bot_cfg = _make_stack(tmp_path)
    ctrl = BotController(storage, kobold, engine, bot_cfg)

    card = CharacterCard(
        name="Aria",
        filename="aria.yaml",
        description="An arcane scholar at the Grand Library.",
        first_mes="\"{{user}}, look at this glowing manuscript!\"",
        alternate_greetings=["\"Welcome back, {{user}}.\""],
        character_book_entries=[
            LorebookEntry(uid="l1", keys=["Starlight Vault"], content="A hidden chamber beneath the library.", position=0, depth=4),
        ],
    )
    sess = storage.create_session(card, kobold.active_model_key, user_name="Player", include_first_mes=True, chat_id=11111)
    assert sess.file_path.parent.name == "11111"

    raw_json_arc = '<think>planning</think>\n' + '{"title": "Echoes of the Starlight Vault", "premise": "Uncovering an ancient cipher hidden inside the Grand Library.", "chapters": [{"name": "Chapter 1: Midnight Cipher", "goal": "Inspect the Starlight Vault entrance", "hooks": "glowing quill, secret door"}, {"name": "Chapter 2: Forgotten Archive", "goal": "Decode the memory crystal", "hooks": "crystal shard"}]}'
    arc = BotController._parse_story_arc_output(raw_json_arc, "Aria")
    assert arc["enabled"] is True
    assert arc["title"] == "Echoes of the Starlight Vault"
    assert len(arc["chapters"]) == 2

    sess.story_arc = arc
    storage.save_session(sess)
    reloaded = storage.load_session(sess.file_path)
    assert reloaded.story_arc["title"] == "Echoes of the Starlight Vault"

    cp_path = storage.create_checkpoint(reloaded, custom_title="arc_cp")
    cp_sess = storage.load_session(cp_path)
    assert cp_sess.story_arc["chapters"][0]["goal"] == "Inspect the Starlight Vault entrance"

    md_path = storage.export_story_markdown(reloaded, card, custom_title="arc_story")
    assert "Echoes of the Starlight Vault" in md_path.read_text(encoding="utf-8")

    assembled = await engine.build_prompt(reloaded, card, pending_user_text="Let us talk.")
    assert "[Story Campaign Outline: Echoes of the Starlight Vault]" in assembled.prompt_text
    assert "[Story Arc Guidance - Chapter 1: Midnight Cipher]" in assembled.prompt_text
    assert "A hidden chamber beneath the library." in assembled.prompt_text

    # Chat ID isolation
    sess2 = storage.create_session(card, kobold.active_model_key, user_name="GroupUser", chat_id=-10022222)
    assert len(storage.list_sessions(chat_id=11111)) == 2  # session + checkpoint
    assert len(storage.list_sessions(chat_id=-10022222)) == 1
    assert storage.get_latest_session_for_char("Aria", chat_id=-10022222).user_name == "GroupUser"


def test_two_level_model_menu(tmp_path: Path) -> None:
    storage, kobold, engine, bot_cfg = _make_stack(tmp_path)
    ctrl = BotController(storage, kobold, engine, bot_cfg)
    rt = ctrl.get_runtime(12345)

    _, fw_markup = ctrl._build_model_framework_menu(rt, live_model=None)
    fw_callbacks = [btn.callback_data for row in fw_markup.inline_keyboard for btn in row]
    assert "mdl_fw:koboldcpp" in fw_callbacks
    assert "mdl_fw:ollama" in fw_callbacks
    assert "mdl_fw:mlx" in fw_callbacks
    assert "mdl_fw:ollama_cloud" in fw_callbacks

    _, oc_markup = ctrl._build_model_list_menu(rt, "ollama_cloud", live_model=None)
    oc_labels = [btn.text for row in oc_markup.inline_keyboard for btn in row]
    assert any("Cloud" in lbl for lbl in oc_labels)


@pytest.mark.asyncio
async def test_audio_adapter_roundtrip(monkeypatch) -> None:
    from core.audio_adapter import AudioAdapter

    base_dir = Path(__file__).resolve().parent.parent
    adapter = AudioAdapter(bot_config={}, base_dir=base_dir)
    cleaned = adapter.clean_text_for_speech("<think>secret</think> *smiles* \"Hello there!\"\n[💡 Speaking Coach: nice!]")
    assert "secret" not in cleaned
    assert "Speaking Coach" not in cleaned
    assert "Hello there!" in cleaned

    model_exists = (adapter.piper_dir / f"{adapter.tts_default_voice}.onnx").exists()
    if not model_exists:
        class _DummyVoice:
            def synthesize_wav(self, text: str, wav_file) -> None:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(22050)
                wav_file.writeframes(b"\x00\x00" * 2205)

        monkeypatch.setattr(adapter, "_get_piper_voice", lambda name: _DummyVoice())

        async def _fake_transcribe(ogg: bytes) -> str:
            return "Good evening, traveler. The stars above Neverland are brilliant tonight."

        monkeypatch.setattr(adapter, "transcribe_ogg", _fake_transcribe)

    sample = "Good evening, traveler. The stars above Neverland are brilliant tonight."
    ogg_bytes = await adapter.synthesize_ogg(sample, char_name="Peter Pan")
    assert ogg_bytes.startswith(b"OggS")
    transcript = await adapter.transcribe_ogg(ogg_bytes)
    assert "neverland" in transcript.lower()
    assert "brilliant" in transcript.lower()

    # Test dual voice segmentation
    dual_sample = '*She walks forward.* "Can you hear me?" *She smiles.*'
    segments = adapter.segment_speech_elements(dual_sample)
    assert len(segments) == 3
    assert segments[0] == ("narration", "She walks forward.")
    assert segments[1] == ("dialogue", "Can you hear me?")
    assert segments[2] == ("narration", "She smiles.")
    dual_ogg = await adapter.synthesize_ogg(dual_sample, char_name="Snow White")
    assert dual_ogg.startswith(b"OggS")
