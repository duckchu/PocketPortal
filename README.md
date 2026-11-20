# PocketPortal

**A standalone, headless SillyTavern-compatible Roleplay Engine for Telegram — built for Apple Silicon Unified Memory, multi-backend local/cloud LLMs, and ComfyUI illustration.**

Unlike Telegram bridge plugins that require running the full Node.js SillyTavern web server in the background, **PocketPortal** is a pure-Python, zero-browser roleplay engine. It packages an **8-stage prompt assembler**, **recursive Lorebook scanning**, **64K KV-cache protection**, **Dynamic Story Arcs**, and **hardware-level memory orchestration** into a lightweight Telegram bot.

---

## Why PocketPortal?

- **Zero Node.js / Browser Overhead**: Runs completely headless as a lightweight Python service—no need to keep a heavy SillyTavern frontend or WebSocket bridge alive.
- **Hardware-Level Unified Memory Manager**: Designed specifically for Mac Apple Silicon (e.g., 32GB unified memory):
  - **Mutual-Exclusion Backend Switching**: Seamlessly switches between **KoboldCpp (Native Metal)**, **Ollama (Local API)**, **MLX (Apple Silicon)**, and **Ollama Cloud**, automatically terminating/unloading the previous backend so models never fight over RAM.
  - **Idle Auto-Unload Watchdog**: Automatically frees model memory after configurable inactivity.
  - **ComfyUI VRAM Handoff**: Automatically frees LLM memory before generating scene illustrations via ComfyUI and calls `/free` afterward to reclaim VRAM.
- **Deep Narrative & Prompt Control**:
  - **8-Stage Prompt Stack** with **64K KV-Cache Protection** (long-term lore/summary stays pinned at Stage 2 while dynamic `@Depth 4` persona/style and `@Depth 2` plot notes interleave cleanly).
  - **Recursive Lorebook Scanning** (`character_book` + external JSON lorebooks) and automatic 85%-context background summarization (`/summarize`).
  - **Dynamic Story Arcs (`/arc`)**: Generates 4-chapter world-building campaigns from character cards & lorebooks while prioritizing player agency—supporting organic divergence and live arc adaptation (`/arc adapt`).
- **Writer-First Creative Loop**:
  - **`➕ Continue`**, **Swipe Rerolls (`⬅️ / ➡️`)**, **Guided Rewrites (`✏️ Rewrite`)**, and **`// OOC` / `(OOC: ...)`** single-turn directorial guidance without polluting chat history.
  - **Drag-and-Drop PNG Character Cards** (SillyTavern V2/V3 metadata parsing) directly in Telegram.
  - **Per-Chat Isolation**: Sessions and checkpoints are isolated by Telegram `chat_id` (`sessions/<chat_id>/`), supporting both private chats and groups.
  - **One-Click Clean Novel Export**: `/save` creates a checkpoint and exports a clean Markdown story with all system/OOC tags stripped.

---

## Quick Start

1. **Clone and install dependencies**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure**:
   ```bash
   cp config/bot_config.example.yaml config/bot_config.yaml
   cp config/models.example.yaml config/models.yaml
   cp config/system_presets.example.yaml config/system_presets.yaml
   ```
   Fill in your `bot_token` and `allowed_user_ids` in `config/bot_config.yaml`, and adjust model paths/names in `config/models.yaml`.

3. **Run**:
   ```bash
   python3 bot.py
   ```

4. **Run Tests**:
   ```bash
   pytest tests/test_all.py -v
   ```

---

## Key Telegram Commands

| Command | Description |
| :--- | :--- |
| `/model` | Two-level Framework (`KoboldCpp` / `Ollama` / `MLX` / `Ollama Cloud`) & model switcher |
| `/char` | Select character card (New Story, World Story Arc Campaign, or Resume) |
| `/arc [theme]` | Dynamic Story Arc console (advance chapters, adapt to plot divergence, toggle) |
| `/world` / `/world_add` | Mount/unmount Lorebooks or add temporary world info entries |
| `/plot` / `/plot_once` | Set persistent `@Depth 2` plot direction or single-turn immediate event |
| `/persona` / `/style` | Dynamically override character persona or output tone (`@Depth 4`) |
| `/ooc` / `/dir` | Out-of-character discussion or director-level scene advance |
| `/retry` / `/edit` / `/rewind` | Reroll with guidance, edit last input, or rewind N turns |
| `/summarize` | Compress oldest turns into a Story Summary |
| `/illustrate` | Generate a scene illustration via ComfyUI |
| `/save` / `/load` | Create checkpoint & export clean Markdown novel, or load saved sessions |
| `/voice` | Voice Interaction & Speaking Coach dashboard (toggle Auto-Voice / Coach Mode) |
| `/status` / `/exit` | View token budget & prompt dashboard, or save & unload model memory |

---

## Voice Interaction & Dual-Voice Narration (STT & TTS)

PocketPortal features a completely local, CPU-based speech engine powered by **faster-whisper** (STT) and **piper-tts** (TTS), maintaining a strict boundary that prevents CMA/VRAM resource contention on edge hardware (such as RK3588 or Apple Silicon):

- **🎙️ Direct Voice Input (STT)**: Send Telegram voice notes or audio messages directly. PocketPortal transcribes them on the CPU via `faster-whisper` and streams your speech into the roleplay session naturally.
- **🔊 Multi-Voice & Dual-Voice Narration (TTS)**:
  - Both character spoken dialogue and world narration/exposition are read aloud.
  - **Narrator & Dialogue Voice Separation**: Spoken dialogue (e.g. `"..."` or `“...”`) is synthesized using the designated character voice (with automatic male/female British voice selection), while narration, scene descriptions, and stage directions are voiced by a distinct British narrator voice (`en_GB-alba-medium` / `en_GB-alan-medium`).
  - Listen on-demand via the inline **`🔊 Speak`** button beneath any message, or enable automatic voice replies via `/voice`.
- **🗣️ Speaking Coach Mode (`/voice`)**:
  - Practice spoken English conversation interactively with your characters.
  - When enabled, characters respond naturally in-character, followed by an unobtrusive `[💡 Speaking Coach: ...]` tip analyzing grammar, word choice, or phrasing nuances. Coach notes are excluded from voice output to preserve dramatic immersion.

---

## Localisation

English is the default interface language. Every user-facing string lives in
`i18n/en.json`; `i18n/zh-TW.json` ships a Traditional Chinese translation.

```bash
export ST_LOCALE=zh-TW   # switch the bot to Traditional Chinese
```

Missing keys fall back to the English source string.

## Character Personas & Illustration Prompts

The illustration module ships **neutral examples only**. Personal characters,
LoRA file names, and prompt libraries are never committed:

```bash
cp illust/personas/example_personas.yaml.example illust/personas/personas.yaml
cp illust/prompts/cloth.example.txt illust/prompts/cloth.txt
```

`illust/prompts/*.txt` and `illust/personas/*.yaml` are git-ignored. Set
`ST_PERSONAS_FILE=/path/to/personas.yaml` to keep them outside the repo.
Only use characters and LoRA models you have the rights to use.

## Example Characters & Lorebooks

PocketPortal ships neutral example character cards (`characters/*.yaml.example`) and lorebooks (`lorebooks/*.json.example`). Copy the examples you want to use:

```bash
cp characters/example_character.yaml.example characters/example_character.yaml
cp lorebooks/example_lorebook.json.example lorebooks/example_lorebook.json
```

Other bundled examples include `little_prince.yaml.example`, `peter_pan.yaml.example`, `snow_white.yaml.example`, and their corresponding `*_lorebook.json.example` files.

## RK3588 NPU (rkllama) Backend

To use [rkllama](https://github.com/NotPunchnox/rkllama) on Rockchip RK3588/RK3576 devices:

1. Point `ollama_url` in `config/bot_config.yaml` to your `rkllama` server (default port `8080`):
   ```yaml
   ollama_url: http://127.0.0.1:8080
   ```
2. Add your `.rkllm` models in `config/models.yaml` with `backend: ollama` and set `context_size: 4096` (or match your `.rkllm` model's `max_context_limit`). See `rkllama-example` in `config/models.example.yaml`.

## Voice Support (STT & TTS)

PocketPortal supports local CPU speech-to-text (`faster-whisper`) and text-to-speech (`piper-tts`) via `ffmpeg` and OGG/Opus.

### Installation

1. Install **Python 3.12** and **ffmpeg** (`onnxruntime` requires Python <= 3.12):
   ```bash
   brew install python@3.12 ffmpeg
   ```
2. Install voice dependencies (pin `av<14` for `faster-whisper` compatibility):
   ```bash
   pip install faster-whisper piper-tts "av<14"
   ```
3. Download Piper voice models (`.onnx` & `.onnx.json`) from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) into `audio_models/piper/`.
4. Optionally specify a character voice in its YAML/JSON card via `tts_voice`:
   ```yaml
   tts_voice: en_GB-cori-medium
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
