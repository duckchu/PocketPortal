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
| `/status` / `/exit` | View token budget & prompt dashboard, or save & unload model memory |

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
