#!/usr/bin/env python3
"""PocketPortal Roleplay Telegram Bot entrypoint."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from telegram import BotCommand
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from core.context_engine import ContextEngine
from core.handlers import BotController
from core.llm_adapter import LLMAdapter
from core.storage import StorageManager

BASE_DIR = Path(__file__).resolve().parent

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logger = logging.getLogger("PocketPortal.Bot")


async def post_init(application: Application) -> None:
    commands = [
        BotCommand("start", "Show help and current status"),
        BotCommand("model", "Switch inference framework and model"),
        BotCommand("char", "Switch character card and bind lorebooks"),
        BotCommand("world", "Mount or unmount world lorebooks"),
        BotCommand("world_add", "Add temporary world lorebook entry"),
        BotCommand("arc", "Dynamic story arc console (generate/advance/adapt)"),
        BotCommand("plot", "Set long-term plot direction (@Depth 2)"),
        BotCommand("plot_once", "Set single-turn immediate event"),
        BotCommand("plot_clear", "Clear plot direction and single-turn event"),
        BotCommand("persona", "Override character persona (@Depth 4)"),
        BotCommand("style", "Override output style and tone (@Depth 4)"),
        BotCommand("reset_trait", "Reset persona and style overrides"),
        BotCommand("ooc", "Out-of-character discussion (not saved to history)"),
        BotCommand("dir", "Director instruction to advance the scene"),
        BotCommand("retry", "Reroll or rewrite with guidance"),
        BotCommand("edit", "Edit last user input and regenerate"),
        BotCommand("rewind", "Rewind last N turns"),
        BotCommand("save", "Create checkpoint and export story Markdown"),
        BotCommand("load", "Load saved session or checkpoint"),
        BotCommand("summarize", "Summarize oldest turns into story summary"),
        BotCommand("illustrate", "Generate scene illustration via ComfyUI"),
        BotCommand("voice", "Toggle auto voice reply and speaking coach"),
        BotCommand("exit", "Save progress and unload model memory"),
        BotCommand("status", "Show system, token budget, and prompt status"),
    ]
    try:
        await application.bot.set_my_commands(commands)
    except Exception as exc:
        logger.warning("Failed to set Telegram bot commands: %s", exc)

    controller = application.bot_data.get("controller")
    if controller is not None:
        if controller.idle_unload_enabled:
            asyncio.create_task(controller.idle_watchdog_loop(application.bot))
            logger.info("Idle auto-unload watchdog started (timeout %ss).", int(controller.idle_unload_seconds))
        else:
            logger.info("Idle auto-unload watchdog disabled.")


def build_application() -> Application:
    storage = StorageManager(BASE_DIR)
    bot_config = storage.load_bot_config()
    models_config = storage.load_models_config()
    presets_config = storage.load_system_presets()

    token = str(bot_config.get("bot_token", "")).strip()
    if not token:
        logger.error(
            "Telegram Bot Token not set! Please fill bot_token in %s/config/bot_config.yaml or set ST_BOT_TOKEN.",
            BASE_DIR,
        )
        sys.exit(1)

    kobold = LLMAdapter(bot_config=bot_config, models_config=models_config, logs_dir=storage.logs_dir)
    engine = ContextEngine(storage=storage, kobold=kobold, bot_config=bot_config, presets_config=presets_config)
    controller = BotController(storage=storage, kobold=kobold, engine=engine, bot_config=bot_config)

    app = (
        Application.builder()
        .token(token)
        .connect_timeout(15.0)
        .read_timeout(30.0)
        .write_timeout(30.0)
        .post_init(post_init)
        .build()
    )
    app.bot_data["controller"] = controller

    app.add_handler(CommandHandler(["start", "help"], controller.cmd_start))
    app.add_handler(CommandHandler("model", controller.cmd_model))
    app.add_handler(CommandHandler("char", controller.cmd_char))
    app.add_handler(CommandHandler("world", controller.cmd_world))
    app.add_handler(CommandHandler("world_add", controller.cmd_world_add))
    app.add_handler(CommandHandler("arc", controller.cmd_arc))
    app.add_handler(CommandHandler("plot", controller.cmd_plot))
    app.add_handler(CommandHandler("plot_once", controller.cmd_plot_once))
    app.add_handler(CommandHandler("plot_clear", controller.cmd_plot_clear))
    app.add_handler(CommandHandler("persona", controller.cmd_persona))
    app.add_handler(CommandHandler("style", controller.cmd_style))
    app.add_handler(CommandHandler("reset_trait", controller.cmd_reset_trait))
    app.add_handler(CommandHandler("ooc", controller.cmd_ooc))
    app.add_handler(CommandHandler("dir", controller.cmd_dir))
    app.add_handler(CommandHandler("retry", controller.cmd_retry))
    app.add_handler(CommandHandler("edit", controller.cmd_edit))
    app.add_handler(CommandHandler("rewind", controller.cmd_rewind))
    app.add_handler(CommandHandler("save", controller.cmd_save))
    app.add_handler(CommandHandler("load", controller.cmd_load))
    app.add_handler(CommandHandler("summarize", controller.cmd_summarize))
    app.add_handler(CommandHandler("illustrate", controller.cmd_illustrate))
    app.add_handler(CommandHandler("voice", controller.cmd_voice))
    app.add_handler(CommandHandler(["exit", "unload"], controller.cmd_exit))
    app.add_handler(CommandHandler("status", controller.cmd_status))

    app.add_handler(CallbackQueryHandler(controller.handle_callback))
    app.add_handler(MessageHandler(filters.Document.IMAGE, controller.handle_document))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, controller.handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), controller.handle_text))

    return app


def main() -> None:
    app = build_application()
    logger.info("PocketPortal Telegram Bot starting polling...")
    app.run_polling()


if __name__ == "__main__":
    main()
