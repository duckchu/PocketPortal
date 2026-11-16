"""Layer 1: Telegram Bot Interface Handlers, Inline Keyboards & Streaming."""

from __future__ import annotations

import asyncio
import hashlib
import html
import io
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest, RetryAfter, TelegramError
from telegram.ext import ContextTypes

from core.context_engine import AssembledPrompt, ContextEngine
from core.llm_adapter import LLMAdapter
from illust.comfy_gen import (
    DEFAULT_NEGATIVE,
    assemble_persona_prompt,
    build_comfyui_workflow,
    resolve_persona_for_character,
)
from core.storage import CharacterCard, SessionState, StorageManager, Turn

logger = logging.getLogger("STLite.Handlers")


def _short_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _esc(val: Any) -> str:
    return html.escape(str(val))


@dataclass
class ChatRuntime:
    chat_id: int
    session: Optional[SessionState] = None
    character: Optional[CharacterCard] = None
    generating: bool = False
    cancel_event: Optional[asyncio.Event] = None
    awaiting_rewrite_guidance: bool = False
    rewrite_target_turn_id: Optional[int] = None
    awaiting_arc_theme: bool = False
    pending_arc_char_filename: Optional[str] = None
    pending_arc_data: Optional[dict[str, Any]] = None
    id_map: dict[str, str] = field(default_factory=dict)

    def register_id(self, prefix: str, full_value: str) -> str:
        sid = _short_id(f"{prefix}:{full_value}")
        key = f"{prefix}:{sid}"
        self.id_map[key] = full_value
        while len(self.id_map) > 500:
            oldest = next(iter(self.id_map))
            self.id_map.pop(oldest, None)
        return sid

    def resolve_id(self, prefix: str, sid: str) -> Optional[str]:
        return self.id_map.get(f"{prefix}:{sid}")


class BotController:
    """Coordinates StorageManager, LLMAdapter, ContextEngine, and Telegram state."""

    def __init__(
        self,
        storage: StorageManager,
        kobold: LLMAdapter,
        engine: ContextEngine,
        bot_config: dict[str, Any],
    ) -> None:
        self.storage = storage
        self.kobold = kobold
        self.engine = engine
        self.bot_config = bot_config
        self.allowed_ids: set[int] = {int(x) for x in (bot_config.get("allowed_user_ids") or [])}
        if not self.allowed_ids:
            logger.warning("allowed_user_ids is empty; all Telegram requests will be denied (fail-closed).")
        self.stream_interval: float = float(bot_config.get("stream_interval_sec", 1.6))
        self.idle_unload_seconds: float = float(bot_config.get("idle_unload_seconds", 300))
        self.idle_unload_enabled: bool = bool(bot_config.get("idle_unload_enabled", True)) and self.idle_unload_seconds > 0
        self.comfyui_url: str = str(bot_config.get("comfyui_url", "http://127.0.0.1:8188")).rstrip("/")
        self.comfyui_checkpoint: str = str(
            bot_config.get("comfyui_checkpoint", "model.safetensors")
        )
        self.comfyui_width: int = int(bot_config.get("comfyui_width", 832))
        self.comfyui_height: int = int(bot_config.get("comfyui_height", 1216))
        self.comfyui_steps: int = int(bot_config.get("comfyui_steps", 26))
        self.comfyui_cfg: float = float(bot_config.get("comfyui_cfg", 5.5))
        self.comfyui_sampler: str = str(bot_config.get("comfyui_sampler", "euler_ancestral"))
        self.comfyui_scheduler: str = str(bot_config.get("comfyui_scheduler", "normal"))
        self.last_active_chat_id: Optional[int] = None
        self.chats: dict[int, ChatRuntime] = {}
        self._summarizing_chats: set[int] = set()
        self._last_chunk_ids: list[int] = []

    def is_authorized(self, user_id: Optional[int]) -> bool:
        if not self.allowed_ids or user_id is None:
            return False
        return int(user_id) in self.allowed_ids

    def get_runtime(self, chat_id: int) -> ChatRuntime:
        self.last_active_chat_id = chat_id
        if chat_id not in self.chats:
            rt = ChatRuntime(chat_id=chat_id)
            recent = self.storage.list_sessions(limit=1, chat_id=chat_id)
            if recent:
                try:
                    sess = self.storage.load_session(recent[0]["path"])
                    char = self.storage.load_character(sess.char_filename or sess.char_name)
                    rt.session = sess
                    rt.character = char
                    if sess.active_model and sess.active_model in self.kobold.models:
                        self.kobold.active_model_key = sess.active_model
                except Exception as exc:
                    logger.warning("Could not auto-restore session: %s", exc)
            self.chats[chat_id] = rt
        return self.chats[chat_id]

    async def _require_auth(self, update: Update) -> bool:
        uid = update.effective_user.id if update.effective_user else None
        if self.is_authorized(uid):
            return True
        msg = "⛔ You are not authorized to use this bot."
        if update.callback_query:
            await update.callback_query.answer(msg, show_alert=True)
        elif update.effective_message:
            await update.effective_message.reply_text(msg)
        return False

    # ------------------------------------------------------------------
    # Keyboards
    # ------------------------------------------------------------------
    @staticmethod
    def swipe_keyboard(turn: Turn) -> InlineKeyboardMarkup:
        total = max(1, len(turn.swipes))
        cur = min(max(0, turn.active_swipe), total - 1) + 1
        tid = turn.turn_id
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("🔄 Reroll", callback_data=f"sw:reroll:{tid}"),
                InlineKeyboardButton(f"⬅️ ({cur}/{total})", callback_data=f"sw:prev:{tid}"),
                InlineKeyboardButton(f"➡️ ({cur}/{total})", callback_data=f"sw:next:{tid}"),
                InlineKeyboardButton("✏️ Rewrite", callback_data=f"sw:guide:{tid}"),
            ],
            [
                InlineKeyboardButton("➕ Continue", callback_data=f"sw:cont:{tid}"),
                InlineKeyboardButton("🎨 Illustrate", callback_data=f"sw:illustrate:{tid}"),
            ],
        ])

    @staticmethod
    def stop_keyboard() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("🛑 Stop generating", callback_data="stop_gen")]
        ])

    def _world_keyboard(self, rt: ChatRuntime) -> InlineKeyboardMarkup:
        all_lbs = self.storage.list_lorebooks()
        active_set = set(rt.session.bound_lorebooks) if rt.session else set()
        rows: list[list[InlineKeyboardButton]] = []
        for lb in all_lbs:
            sid = rt.register_id("lb", lb)
            icon = "✅" if lb in active_set else "⬜"
            rows.append([InlineKeyboardButton(f"{icon} {lb}", callback_data=f"world_toggle:{sid}")])
        rows.append([InlineKeyboardButton("✔️ Done", callback_data="world_done")])
        return InlineKeyboardMarkup(rows)

    @staticmethod
    def _trim_repetition_loop(text: str) -> tuple[str, bool]:
        """Detect consecutive repeating tail blocks or 3rd occurrence of an 18-char phrase and trim."""
        n = len(text)
        if n < 36:
            return text, False
        for seg_len in range(16, min(140, n // 2) + 1):
            tail = text[-seg_len:]
            if not tail.strip():
                continue
            prev = text[-2 * seg_len : -seg_len]
            if tail == prev:
                return text[:-seg_len], True
        if n >= 54:
            probe = text[-18:]
            if len(probe.strip()) >= 10 and text[:-18].count(probe) >= 2:
                return text[:-18], True
        return text, False

    async def _safe_edit_text(
        self,
        bot: Any,
        chat_id: int,
        message_id: int,
        text: str,
        reply_markup: Optional[InlineKeyboardMarkup] = None,
        use_html: bool = True,
        already_html: bool = False,
    ) -> None:
        if not text.strip():
            text = "..."
        if len(text) > 4000:
            text = text[:3990] + "\n…"
        if already_html:
            formatted = text
            parse_mode: Optional[str] = ParseMode.HTML
        elif use_html:
            formatted = self.engine.format_for_telegram_html(text)
            parse_mode = ParseMode.HTML
        else:
            formatted = text
            parse_mode = None

        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=formatted,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
        except RetryAfter as exc:
            await asyncio.sleep(float(exc.retry_after) + 0.5)
        except BadRequest as exc:
            msg = str(exc).lower()
            if "message is not modified" in msg:
                return
            if parse_mode and ("parse" in msg or "tag" in msg or "entities" in msg):
                try:
                    await bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=text,
                        reply_markup=reply_markup,
                    )
                except TelegramError:
                    pass
            else:
                logger.debug("edit_message_text BadRequest: %s", exc)
        except TelegramError as exc:
            logger.debug("edit_message_text error: %s", exc)

    async def _safe_send_html(
        self,
        message: Any,
        text: str,
        reply_markup: Optional[InlineKeyboardMarkup] = None,
        already_html: bool = False,
    ) -> Any:
        chunks = self._split_text(text, 3800)
        sent = None
        sent_ids: list[int] = []
        for idx, chunk in enumerate(chunks):
            kb = reply_markup if idx == len(chunks) - 1 else None
            formatted = chunk if already_html else self.engine.format_for_telegram_html(chunk)
            try:
                sent = await message.reply_text(formatted, parse_mode=ParseMode.HTML, reply_markup=kb)
            except BadRequest:
                sent = await message.reply_text(chunk, reply_markup=kb)
            if sent is not None and getattr(sent, "message_id", None):
                sent_ids.append(int(sent.message_id))
        self._last_chunk_ids = sent_ids
        return sent

    @staticmethod
    def _split_text(text: str, limit: int = 3800) -> list[str]:
        if len(text) <= limit:
            return [text]
        parts: list[str] = []
        rest = text
        while len(rest) > limit:
            cut = rest.rfind("\n", 0, limit)
            if cut < limit // 2:
                cut = limit
            parts.append(rest[:cut].strip())
            rest = rest[cut:].strip()
        if rest:
            parts.append(rest)
        return parts

    # ------------------------------------------------------------------
    # Generation & Streaming Core
    # ------------------------------------------------------------------
    async def _run_generation(
        self,
        rt: ChatRuntime,
        bot: Any,
        chat_id: int,
        target_message: Any,
        assembled: AssembledPrompt,
        edit_in_place: bool = False,
        target_msg_id: Optional[int] = None,
        seed: Optional[int] = None,
        temp_override: Optional[float] = None,
        is_ooc: bool = False,
        target_turn: Optional[Turn] = None,
        continue_append: bool = False,
    ) -> None:
        if rt.generating:
            await self.kobold.abort_generation()
            if rt.cancel_event:
                rt.cancel_event.set()
            await asyncio.sleep(0.3)

        rt.generating = True
        cancel_event = asyncio.Event()
        rt.cancel_event = cancel_event

        model_info = self.kobold.get_model_info(self.kobold.active_model_key)
        sampling = dict(model_info.get("sampling") or {})

        base_prefix = target_turn.text.rstrip() if (continue_append and target_turn is not None) else ""

        if edit_in_place and target_turn is not None:
            old_msg_ids = target_turn.get_message_ids()
            if not target_msg_id and old_msg_ids:
                target_msg_id = old_msg_ids[0]
            for extra_mid in old_msg_ids[1:]:
                try:
                    await bot.delete_message(chat_id=chat_id, message_id=extra_mid)
                except Exception:
                    pass

        if edit_in_place and target_msg_id:
            msg_id = target_msg_id
            status_text = "⏳ Continuing the reply..." if continue_append else "⏳ Regenerating reply..."
            await self._safe_edit_text(
                bot, chat_id, msg_id, status_text, reply_markup=self.stop_keyboard(), use_html=False
            )
        else:
            sent = await target_message.reply_text("⏳ Generating reply...", reply_markup=self.stop_keyboard())
            msg_id = sent.message_id

        raw_buffer = ""
        last_edit_ts = 0.0
        last_sent_text = ""
        char_name = rt.character.name if rt.character else ""
        user_name = rt.session.user_name if rt.session else "Player"

        try:
            if self.kobold.get_backend() == "koboldcpp" and not await self.kobold.get_loaded_model_from_api():
                await self._safe_edit_text(
                    bot, chat_id, msg_id, "⏳ Model was unloaded; reloading...", reply_markup=None, use_html=False
                )
                await self.kobold.ensure_running()
                await self._safe_edit_text(
                    bot, chat_id, msg_id, "⏳ Model loaded; generating reply...", reply_markup=self.stop_keyboard(), use_html=False
                )

            async for token in self.kobold.generate_stream(
                prompt=assembled.prompt_text,
                sampling_params=sampling,
                stop_sequences=assembled.stop_sequences,
                max_context=assembled.max_context,
                cancel_event=cancel_event,
                seed=seed,
                temp_override=temp_override,
            ):
                raw_buffer += token
                trimmed, looped = self._trim_repetition_loop(raw_buffer)
                if looped:
                    raw_buffer = trimmed
                    cancel_event.set()
                    break
                now = asyncio.get_running_loop().time()
                if now - last_edit_ts >= self.stream_interval:
                    preview = self.engine.post_process_output(raw_buffer, char_name, user_name, is_preview=True)
                    if preview and preview != last_sent_text:
                        if continue_append and base_prefix:
                            sep = "\n\n" if raw_buffer.startswith("\n") else ""
                            full_preview = f"{base_prefix}{sep}{preview}"
                        else:
                            full_preview = preview
                        display = full_preview if not is_ooc else f"💬 [OOC reply]\n\n{full_preview}"
                        if len(display) > 3800:
                            display = (
                                f"⏳ [Long generation in progress, {len(full_preview):,} characters so far...]\n\n…"
                                + display[-3650:]
                            )
                        await self._safe_edit_text(
                            bot, chat_id, msg_id, display, reply_markup=self.stop_keyboard(), use_html=False
                        )
                        last_sent_text = preview
                        last_edit_ts = now

            final_text = self.engine.post_process_output(raw_buffer, char_name, user_name, is_preview=False)
            if continue_append and base_prefix:
                if final_text:
                    sep = "\n\n" if raw_buffer.startswith("\n") else ""
                    final_text = f"{base_prefix}{sep}{final_text}"
                else:
                    final_text = base_prefix
            elif not final_text:
                final_text = "(The model returned no valid content, or generation was interrupted.)"

            if is_ooc:
                display = f"💬 [OOC reply]\n\n{final_text}"
                await self._safe_edit_text(bot, chat_id, msg_id, display, reply_markup=None, use_html=True)
                return

            if rt.session:
                if target_turn is not None:
                    if continue_append:
                        target_turn.set_text(final_text)
                    else:
                        target_turn.add_swipe(final_text)
                    target_turn.metadata.pop("token_cache", None)
                    target_turn.set_message_ids([msg_id])
                    turn = target_turn
                else:
                    turn = rt.session.append_turn(
                        role="assistant",
                        text=final_text,
                        telegram_message_id=msg_id,
                        metadata={"triggered_lore": assembled.triggered_lore_keys},
                    )

                actual_prompt_tokens = assembled.prompt_tokens
                if self.kobold.get_backend() == "ollama" and self.kobold.last_prompt_eval_count:
                    actual_prompt_tokens = self.kobold.last_prompt_eval_count

                rt.session.last_triggered_lore = list(assembled.triggered_lore_keys)
                rt.session.last_token_usage = {
                    "prompt_tokens": actual_prompt_tokens,
                    "max_context": assembled.max_context,
                }
                if rt.session.plot_once:
                    rt.session.plot_once = ""
                self.storage.save_session(rt.session)
                kb = self.swipe_keyboard(turn)
            else:
                turn = None
                kb = None

            if len(final_text) <= 3800:
                await self._safe_edit_text(bot, chat_id, msg_id, final_text, reply_markup=kb, use_html=True)
            else:
                chunks = self._split_text(final_text, 3800)
                await self._safe_edit_text(bot, chat_id, msg_id, chunks[0], reply_markup=None, use_html=True)
                sent_ids = [msg_id]
                for idx, chunk in enumerate(chunks[1:], start=1):
                    chunk_kb = kb if idx == len(chunks) - 1 else None
                    sent_extra = await bot.send_message(
                        chat_id=chat_id,
                        text=self.engine.format_for_telegram_html(chunk),
                        parse_mode=ParseMode.HTML,
                        reply_markup=chunk_kb,
                    )
                    sent_ids.append(sent_extra.message_id)
                if rt.session and turn is not None:
                    turn.set_message_ids(sent_ids)
                    self.storage.save_session(rt.session)

            if rt.session and assembled.max_context > 0:
                used_tokens = rt.session.last_token_usage.get("prompt_tokens", 0)
                unsummarized = [t for t in rt.session.turns if t.turn_id > rt.session.summary_upto_turn_id]
                if used_tokens >= int(assembled.max_context * 0.85) and len(unsummarized) > 12:
                    asyncio.create_task(self._auto_summarize_background(bot, chat_id, rt))

        except Exception as exc:
            logger.exception("Generation failed: %s", exc)
            await self._safe_edit_text(
                bot,
                chat_id,
                msg_id,
                f"❌ Generation failed: {exc}\nCheck that KoboldCpp is running (use /model to load a model).",
                reply_markup=None,
                use_html=False,
            )
        finally:
            rt.generating = False
            rt.cancel_event = None
            self.kobold.touch_activity()

    # ------------------------------------------------------------------
    # Command Handlers
    # ------------------------------------------------------------------
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        char_info = rt.character.name if rt.character else "Not selected (use /char)"
        model_info = self.kobold.models.get(self.kobold.active_model_key, {}).get("display_name", self.kobold.active_model_key)
        text = (
            "🎭 <b>PocketPortal Telegram Roleplay Bot</b>\n\n"
            f"• Current character: {_esc(char_info)}\n"
            f"• Default model: {_esc(model_info)}\n\n"
            "<b>Main commands:</b>\n"
            "/model — Switch GGUF model and instruct template\n"
            "/char — Switch character card (auto-binds lorebooks)\n"
            "/world — Mount or unmount lorebooks\n"
            "/world_add &lt;keywords&gt; &lt;content&gt; — Add a temporary lore entry\n"
            "/arc [theme] — Generate or view the story arc (advance or adapt it to the plot)\n"
            "/plot &lt;outline&gt; | /plot_once &lt;single-turn event&gt; | /plot_clear — Plot direction (@Depth 2)\n"
            "/persona &lt;trait&gt; | /style &lt;tone&gt; | /reset_trait — Runtime persona/style override (@Depth 4)\n"
            "/ooc &lt;question&gt; — Out-of-character chat (not saved to history)\n"
            "/dir &lt;instruction&gt; — Director instruction to advance the scene\n"
            "/retry [direction] — Reroll in place / guided rewrite\n"
            "/edit &lt;new line&gt; — Edit your last input and regenerate\n"
            "/rewind &lt;N&gt; — Rewind the last N turns\n"
            "/summarize — Compress the oldest turns into a recap\n"
            "/illustrate [hint] — Generate a ComfyUI illustration of the current scene\n"
            "/save [title] | /load — Checkpoint, export novel Markdown, or load a save\n"
            "/exit — Save progress and free model memory now (idle auto-unload applies)\n"
            "/status — Show model, token budget, and prompt injection dashboard\n\n"
            "💡 Reply buttons support <b>➕ Continue</b>, reroll, swipe left/right, and illustrations. Drop a <b>PNG character card</b> into the chat to import it.\n"
            "💡 Append <code>(OOC: instruction)</code> or <code>// instruction</code> to a message for single-turn out-of-character guidance without polluting history."
        )
        await self._safe_send_html(update.effective_message, text, already_html=True)

    @staticmethod
    def _framework_label(backend: str) -> str:
        labels = {
            "koboldcpp": "⚡ KoboldCpp (Native Metal)",
            "ollama": "🦙 Ollama (Local API)",
            "mlx": "🍎 MLX (Apple Silicon)",
            "ollama_cloud": "☁️ Ollama Cloud",
        }
        return labels.get(backend, f"🔌 {backend}")

    def _build_model_framework_menu(
        self, rt: ChatRuntime, live_model: Optional[str]
    ) -> tuple[str, InlineKeyboardMarkup]:
        models = self.kobold.list_models()
        grouped: dict[str, list[str]] = {}
        for key, info in models.items():
            fw = str(info.get("backend", "koboldcpp")).lower()
            grouped.setdefault(fw, []).append(key)

        active_key = self.kobold.active_model_key
        active_info = models.get(active_key or "", {})
        active_fw = str(active_info.get("backend", "")).lower() if active_info else ""

        ordered_fws = [fw for fw in ("koboldcpp", "ollama", "mlx", "ollama_cloud") if fw in grouped]
        for fw in grouped:
            if fw not in ordered_fws:
                ordered_fws.append(fw)

        rows: list[list[InlineKeyboardButton]] = []
        for fw in ordered_fws:
            mark = "✅ " if active_fw == fw else ""
            label = self._framework_label(fw)
            count = len(grouped[fw])
            rows.append([
                InlineKeyboardButton(f"{mark}{label} ({count} models)", callback_data=f"mdl_fw:{fw}")
            ])

        if live_model or active_key:
            rows.append([InlineKeyboardButton("🧹 Unload Model Memory", callback_data="unload_model")])

        lines = ["🧠 Select an Inference Framework:"]
        if active_info:
            lines.append(f"• Active Config: {active_info.get('display_name', active_key)}")
        if live_model:
            lines.append(f"• Running Model: {live_model}")
        else:
            lines.append("• No running model detected")
        return "\n".join(lines), InlineKeyboardMarkup(rows)

    def _build_model_list_menu(
        self, rt: ChatRuntime, backend: str, live_model: Optional[str]
    ) -> tuple[str, InlineKeyboardMarkup]:
        models = self.kobold.list_models()
        rows: list[list[InlineKeyboardButton]] = []
        for key, info in models.items():
            fw = str(info.get("backend", "koboldcpp")).lower()
            if fw != backend:
                continue
            sid = rt.register_id("mdl", key)
            is_active = key == self.kobold.active_model_key
            mark = "✅ " if is_active else ""
            vram = info.get("vram_gb")
            name = info.get("display_name", key)
            suffix = " (☁️ Cloud)" if (fw == "ollama_cloud" or vram in (None, 0, "cloud")) else f" (~{vram}GB)"
            rows.append([InlineKeyboardButton(f"{mark}{name}{suffix}", callback_data=f"mdl:{sid}")])

        rows.append([InlineKeyboardButton("⬅️ Back to Frameworks", callback_data="mdl_fw_back")])
        fw_label = self._framework_label(backend)
        status_line = f"Running Model: {live_model}" if live_model else "No running model detected"
        text = f"🧠 {fw_label}\nSelect a model to load:\n{status_line}"
        return text, InlineKeyboardMarkup(rows)

    async def cmd_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        live_model = await self.kobold.get_loaded_model_from_api()
        text, markup = self._build_model_framework_menu(rt, live_model)
        await update.effective_message.reply_text(text, reply_markup=markup)

    async def cmd_char(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        chars = self.storage.list_characters()
        if not chars:
            await update.effective_message.reply_text("⚠️ No character cards found in characters/.")
            return

        rows: list[list[InlineKeyboardButton]] = []
        for c in chars:
            sid = rt.register_id("chr", c.filename)
            mark = "✅ " if rt.character and rt.character.filename == c.filename else ""
            lb_names = ", ".join(lb.removesuffix(".json") for lb in c.bound_lorebooks)
            lb_hint = f" [{lb_names}]" if lb_names else ""
            rows.append([InlineKeyboardButton(f"{mark}{c.name}{lb_hint}", callback_data=f"chr_pick:{sid}")])

        await update.effective_message.reply_text(
            "🎭 Select a Character Card:",
            reply_markup=InlineKeyboardMarkup(rows),
        )

    async def cmd_world(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        if not rt.session:
            await update.effective_message.reply_text("⚠️ Load a character story first with /char.")
            return
        await update.effective_message.reply_text(
            "🌍 Tap a button to switch the lorebooks mounted to this story:",
            reply_markup=self._world_keyboard(rt),
        )

    async def cmd_world_add(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        if not rt.session:
            await update.effective_message.reply_text("⚠️ Load a character story first with /char.")
            return
        raw_arg = " ".join(context.args).strip()
        if not raw_arg or " " not in raw_arg:
            await update.effective_message.reply_text("Usage: /world_add <keywords,comma,separated> <content>")
            return
        keys_str, content = raw_arg.split(" ", 1)
        keys = [k.strip() for k in re.split(r"[,，、]+", keys_str) if k.strip()]
        if not keys or not content.strip():
            await update.effective_message.reply_text("Usage: /world_add <keywords> <content>")
            return
        rt.session.add_temp_lore(keys=keys, content=content.strip())
        self.storage.save_session(rt.session)
        await update.effective_message.reply_text(
            f"✅ Temporary lore entry added\n• Keywords: {', '.join(keys)}\n• Content: {content.strip()}"
        )

    # ------------------------------------------------------------------
    # Dynamic Story Arc (/arc & /char Campaign Mode)
    # ------------------------------------------------------------------
    def _collect_world_context_for_arc(
        self, character: CharacterCard, session: Optional[SessionState] = None
    ) -> str:
        parts: list[str] = [f"Character: {character.name}"]
        if character.description.strip():
            parts.append(f"Description:\n{character.description.strip()[:900]}")
        if character.personality.strip():
            parts.append(f"Personality:\n{character.personality.strip()[:500]}")
        if character.scenario.strip():
            parts.append(f"Scenario:\n{character.scenario.strip()[:600]}")

        lb_names = list(session.bound_lorebooks) if session else list(character.bound_lorebooks)
        lore_entries = []
        for lb in lb_names:
            try:
                lore_entries.extend(self.storage.load_lorebook(lb))
            except Exception:
                continue
        lore_entries.extend(character.character_book_entries)
        lore_entries.sort(key=lambda e: (1 if e.constant else 0, e.insertion_order), reverse=True)

        lore_snippets: list[str] = []
        total_lore_chars = 0
        for entry in lore_entries:
            if entry.disable or not entry.content.strip():
                continue
            label = entry.comment or (", ".join(entry.keys[:3]) if entry.keys else "Lore entry")
            snippet = f"- [{label}] {entry.content.strip()[:260]}"
            if total_lore_chars + len(snippet) > 1800:
                break
            lore_snippets.append(snippet)
            total_lore_chars += len(snippet)
            if len(lore_snippets) >= 12:
                break

        if lore_snippets:
            parts.append("Key lore settings (Lorebook):\n" + "\n".join(lore_snippets))
        return "\n\n".join(parts)

    @staticmethod
    def _parse_story_arc_output(
        raw_text: str, character_name: str, theme_hint: str = ""
    ) -> dict[str, Any]:
        cleaned = (raw_text or "").strip()
        cleaned = re.sub(r"(?is)<think>[\s\S]*?</think>", "", cleaned)
        cleaned = re.sub(r"(?is)<\|channel>thought[\s\S]*?<channel\|>", "", cleaned)
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()

        parsed: Optional[dict[str, Any]] = None
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if m:
            try:
                cand = json.loads(m.group(0))
                if isinstance(cand, dict) and (cand.get("chapters") or cand.get("title")):
                    parsed = cand
            except Exception:
                parsed = None

        if parsed is not None:
            chapters_raw = parsed.get("chapters") or []
            chapters: list[dict[str, str]] = []
            for idx, ch in enumerate(chapters_raw[:6]):
                if isinstance(ch, dict):
                    chapters.append({
                        "name": str(ch.get("name") or ch.get("title") or f"Chapter {idx + 1}").strip(),
                        "goal": str(ch.get("goal") or ch.get("summary") or ch.get("description") or "").strip(),
                        "hooks": str(ch.get("hooks") or ch.get("clues") or "").strip(),
                    })
                elif isinstance(ch, str) and ch.strip():
                    chapters.append({"name": f"Chapter {idx + 1}", "goal": ch.strip(), "hooks": ""})
            if chapters:
                return {
                    "enabled": True,
                    "title": str(parsed.get("title") or f"{character_name}'s Lorebook Adventure").strip(),
                    "premise": str(parsed.get("premise") or theme_hint or "An immersive arc built on the lore.").strip(),
                    "current_chapter": int(parsed.get("current_chapter", 0) or 0),
                    "chapters": chapters,
                    "divergence_note": str(parsed.get("divergence_note") or "").strip(),
                }

        # Fallback Markdown / text parser when local models return prose instead of JSON
        title_match = re.search(r"(?:(?:Title|Arc Name))\\s*[:：]\\s*(.+)", cleaned, flags=re.IGNORECASE)
        premise_match = re.search(r"(?:(?:Premise|Synopsis|Background|Setting))\\s*[:：]\\s*(.+)", cleaned, flags=re.IGNORECASE)
        ch_matches = re.findall(
            r"(?:^|\n)\s*(?:[-*#]*\s*)?((?:Chapter|Part|Act)\s*\d+[^\n:]*)\s*[:：]\s*(.+)",
            cleaned,
            flags=re.IGNORECASE,
        )
        chapters = []
        for idx, (cname, cgoal) in enumerate(ch_matches[:4]):
            chapters.append({"name": cname.strip(), "goal": cgoal.strip(), "hooks": ""})
        if not chapters:
            chapters = [
                {"name": "Chapter 1: Prologue and Disturbance", "goal": f"Notice an anomalous sign in {character_name}'s everyday world and make first contact.", "hooks": "mysterious clue, first anomaly"},
                {"name": "Chapter 2: Exploration and Depth", "goal": "Follow the clues into the core locations of the world; trust and rapport build up.", "hooks": "key location exploration, hidden secrets surface"},
                {"name": "Chapter 3: Turning Point and Crisis", "goal": "Hit an unexpected conflict or outside obstacle that tests both sides' rapport and choices.", "hooks": "major choice, direct crisis confrontation"},
                {"name": "Chapter 4: Climax and Aftermath", "goal": "Reach the truth and climax, close out based on prior choices, and open a new normal.", "hooks": "truth revealed, ending branch"},
            ]
        return {
            "enabled": True,
            "title": title_match.group(1).strip().strip("《》「」") if title_match else f"{character_name} - Lorebook Arc",
            "premise": premise_match.group(1).strip() if premise_match else (theme_hint or f"An immersive story set inside {character_name}'s world."),
            "current_chapter": 0,
            "chapters": chapters,
            "divergence_note": "",
        }

    async def _run_raw_llm_prompt(self, user_prompt: str, max_length: int = 1200, temp: float = 0.7) -> str:
        await self.kobold.ensure_running()
        tpl_name = self.kobold.get_active_instruct_template()
        tpl = self.engine._get_template(tpl_name)
        assistant_start = str(
            tpl.get("assistant_prefill") or tpl.get("assistant_prefix", "<|im_start|>assistant\n")
        )
        full_prompt = (
            str(tpl.get("bos_token", ""))
            + self.engine._wrap_block("user", user_prompt, tpl)
            + assistant_start
        )
        model_info = self.kobold.get_model_info(self.kobold.active_model_key)
        sampling = dict(model_info.get("sampling") or {})
        sampling["max_length"] = max_length
        chunks: list[str] = []
        async for tok in self.kobold.generate_stream(
            prompt=full_prompt,
            sampling_params=sampling,
            stop_sequences=list(tpl.get("stop_sequences") or ["<|im_end|>"]),
            max_context=int(model_info.get("context_size", 65536)),
            temp_override=temp,
        ):
            chunks.append(tok)
        return "".join(chunks)

    async def _generate_story_arc_json(
        self,
        character: CharacterCard,
        user_name: str,
        theme_hint: str = "",
        session: Optional[SessionState] = None,
    ) -> dict[str, Any]:
        world_ctx = self._collect_world_context_for_arc(character, session)
        theme_line = f"Requested arc style/theme: {theme_hint}" if theme_hint else "Freely devise the most compelling mainline event from the character and lore."
        prompt = (
            "You are a writer skilled at structuring roleplay worldbuilding. Based on the character and lore settings below, "
            "Design an immersive interactive story arc in 4 chapters (setup, development, turn, climax) for the player \"{{user}}\" and the character \"{{char}}\".\n"
            "Notes:\n"
            "1. Stay strictly faithful to lore terminology, magic/tech rules, and character personalities.\n"
            "2. For each chapter, only design the background event and foreshadowing goal. Leave ample room for player choices to influence or rewrite the direction.\n"
            "3. Output raw JSON only. No extra prose or markdown fences. JSON format:\n"
            '{"title":"Arc title","premise":"Core premise and motivation in under 50 words","chapters":['
            "{\"name\":\"Chapter 1: ...\",\"goal\":\"this chapter event and foreshadowing\",\"hooks\":\"hook one, hook two\"},"
            '{"name":"Chapter 2: ...","goal":"...","hooks":"..."},'
            '{"name":"Chapter 3: ...","goal":"...","hooks":"..."},'
            '{"name":"Chapter 4: ...","goal":"...","hooks":"..."}]}\n\n'
            f"{theme_line}\n\n"
            f"{self.engine.replace_macros(world_ctx, character.name, user_name)}"
        )
        prompt = self.engine.replace_macros(prompt, character.name, user_name)
        raw = await self._run_raw_llm_prompt(prompt, max_length=1400, temp=0.75)
        arc = self._parse_story_arc_output(raw, character.name, theme_hint)
        if self.engine._opencc is not None:
            try:
                arc["title"] = self.engine._opencc.convert(arc["title"])
                arc["premise"] = self.engine._opencc.convert(arc["premise"])
                for ch in arc["chapters"]:
                    ch["name"] = self.engine._opencc.convert(ch["name"])
                    ch["goal"] = self.engine._opencc.convert(ch["goal"])
                    ch["hooks"] = self.engine._opencc.convert(ch["hooks"])
            except Exception:
                pass
        return arc

    async def _generate_arc_opening_scene(
        self,
        character: CharacterCard,
        user_name: str,
        arc: dict[str, Any],
        session: SessionState,
    ) -> str:
        ch1 = (arc.get("chapters") or [{}])[0]
        world_ctx = self._collect_world_context_for_arc(character, session)
        prompt = (
            f"From the perspective and voice of \"{character.name}\", write the interactive opening of act 1 for the arc below (250-400 words).\n"
            "Writing rules:\n"
            "- Narration uses no brackets or asterisks. Dialogue goes in double quotes \"\". Inner thoughts go in parentheses ().\n"
            "- Naturally establish the chapter 1 setting and foreshadowing, and leave a natural entry point for the player (" + user_name + ") to respond or act.\n"
            "- Never decide the player's lines or actions for them.\n\n"
            f"Arc title: \"{arc.get('title', '')}\"\n"
            f"Core premise: {arc.get('premise', '')}\n"
            f"Chapter 1: {ch1.get('name', '')} — {ch1.get('goal', '')} (hooks: {ch1.get('hooks', '')})\n\n"
            f"{self.engine.replace_macros(world_ctx, character.name, user_name)}"
        )
        raw = await self._run_raw_llm_prompt(prompt, max_length=900, temp=0.85)
        cleaned = self.engine.post_process_output(raw, character.name, user_name)
        return cleaned or self.engine.replace_macros(character.first_mes, character.name, user_name)

    async def _adapt_story_arc(self, rt: ChatRuntime, custom_direction: str = "") -> dict[str, Any]:
        assert rt.session is not None and rt.character is not None
        sess = rt.session
        arc = dict(sess.story_arc or {})
        chapters = list(arc.get("chapters") or [])
        cur_idx = max(0, min(int(arc.get("current_chapter", 0)), len(chapters) - 1)) if chapters else 0

        recent_lines = []
        for t in sess.turns[-8:]:
            speaker = sess.user_name if t.role == "user" else sess.char_name
            recent_lines.append(f"{speaker}: {t.text.strip()[:350]}")
        transcript = "\n".join(recent_lines)
        summary_block = f"[Recap]\n{sess.story_summary}\n\n" if sess.story_summary else ""
        dir_line = f"Direction the user wants: {custom_direction}\n" if custom_direction else ""
        cur_json = json.dumps(arc, ensure_ascii=False)

        prompt = (
            "You are a story writer skilled at tracking player choices dynamically.\n"
            "Compare the current arc outline against what actually happened in the conversation. Follow the player's real decisions and their cascading effects, "
            "Update the current chapter progress (current_chapter) and record divergences or new branches (divergence_note), "
            "Keep completed chapters intact, then re-plan the current and future chapters so the new direction becomes the new mainline.\n"
            "Output raw JSON only (fields: title, premise, current_chapter, divergence_note, chapters array):\n\n"
            f"Current arc JSON:\n{cur_json}\n\n"
            f"{dir_line}{summary_block}[Recent transcript]\n{transcript}"
        )
        raw = await self._run_raw_llm_prompt(prompt, max_length=1400, temp=0.65)
        updated = self._parse_story_arc_output(raw, rt.character.name, arc.get("premise", ""))
        if not updated.get("title") and arc.get("title"):
            updated["title"] = arc["title"]
        if not updated.get("divergence_note"):
            updated["divergence_note"] = custom_direction or "Remaining chapters adapted to the latest direction."
        updated["enabled"] = True
        sess.story_arc = updated
        self.storage.save_session(sess)
        return updated

    @staticmethod
    def _format_arc_display(arc: dict[str, Any]) -> tuple[str, InlineKeyboardMarkup]:
        enabled = bool(arc.get("enabled", True))
        status_tag = "🟢 Active" if enabled else "⏸️ Paused (Freeform)"
        chapters = list(arc.get("chapters") or [])
        cur_idx = max(0, min(int(arc.get("current_chapter", 0)), len(chapters) - 1)) if chapters else 0

        lines = [
            f"🗺️ <b>Story Arc: { _esc(arc.get('title', 'Untitled Campaign')) }</b>（{status_tag}）",
            f"• <b>Premise</b>: {_esc(arc.get('premise', '(none)'))}",
        ]
        if arc.get("divergence_note"):
            lines.append(f"• <b>Branching</b>: {_esc(arc['divergence_note'])}")
        lines.append("")
        for idx, ch in enumerate(chapters):
            if idx < cur_idx:
                icon = "✅"
            elif idx == cur_idx:
                icon = "📍"
            else:
                icon = "⏳"
            ch_name = ch.get("name", f"Chapter {idx + 1}")
            ch_goal = ch.get("goal", "")
            ch_hooks = ch.get("hooks", "")
            lines.append(f"{icon} <b>{_esc(ch_name)}</b>")
            if ch_goal:
                lines.append(f"   {_esc(ch_goal)}")
            if idx == cur_idx and ch_hooks:
                lines.append(f"   <i>Hooks: { _esc(ch_hooks) }</i>")

        toggle_label = "⏸️ Pause Arc" if enabled else "▶️ Resume Arc"
        kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("⏮️ Prev Chapter", callback_data="arc_ctl:prev"),
                InlineKeyboardButton("⏭️ Next Chapter", callback_data="arc_ctl:next"),
            ],
            [
                InlineKeyboardButton("🔀 Adapt Arc", callback_data="arc_ctl:adapt"),
                InlineKeyboardButton("🎲 Regenerate Arc", callback_data="arc_ctl:regen"),
            ],
            [
                InlineKeyboardButton(toggle_label, callback_data="arc_ctl:toggle"),
                InlineKeyboardButton("▶️ Start Arc", callback_data="arc_ctl:play"),
            ],
        ])
        return "\n".join(lines), kb

    async def cmd_arc(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        rt.awaiting_rewrite_guidance = False
        rt.rewrite_target_turn_id = None
        if not rt.session or not rt.character:
            await update.effective_message.reply_text("⚠️ Load a character with /char first (you can then pick \"🗺️ Generate story arc\").")
            return

        arg_text = " ".join(context.args).strip()
        if not arg_text:
            arc = rt.session.story_arc
            if arc and (arc.get("title") or arc.get("chapters")):
                text, kb = self._format_arc_display(arc)
                await self._safe_send_html(update.effective_message, text, reply_markup=kb, already_html=True)
            else:
                kb = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🎲 Generate a 4-chapter arc from the lore", callback_data="arc_ctl:regen")],
                    [InlineKeyboardButton("✏️ Enter a custom arc theme", callback_data="arc_ctl:custom_prompt")],
                ])
                await update.effective_message.reply_text(
                    f"🗺️ No story arc exists yet for the conversation with <b>{rt.character.name}</b>.\n"
                    "Tap a button below to generate an arc from the character and lorebooks, or send: <code>/arc your theme</code>.",
                    parse_mode=ParseMode.HTML,
                    reply_markup=kb,
                )
            return

        if arg_text.lower().startswith("adapt"):
            direction = arg_text[5:].strip()
            if not rt.session.story_arc:
                await update.effective_message.reply_text("ℹ️ No story arc yet. Generate one first with /arc.")
                return
            status_msg = await update.effective_message.reply_text("🔀 Rebuilding the remaining chapters from the current direction...")
            try:
                updated = await self._adapt_story_arc(rt, custom_direction=direction)
                text, kb = self._format_arc_display(updated)
                await self._safe_edit_text(
                    context.bot, update.effective_chat.id, status_msg.message_id, text, reply_markup=kb, already_html=True
                )
            except Exception as exc:
                await status_msg.edit_text(f"❌ Failed to rebuild the story arc: {exc}")
            return

        status_msg = await update.effective_message.reply_text("🗺️ Generating a story arc from the character and lorebooks...")
        try:
            arc = await self._generate_story_arc_json(
                character=rt.character,
                user_name=rt.session.user_name,
                theme_hint=arg_text,
                session=rt.session,
            )
            rt.session.story_arc = arc
            self.storage.save_session(rt.session)
            text, kb = self._format_arc_display(arc)
            await self._safe_edit_text(
                context.bot, update.effective_chat.id, status_msg.message_id, text, reply_markup=kb, already_html=True
            )
        except Exception as exc:
            await status_msg.edit_text(f"❌ Failed to generate the story arc: {exc}")

    async def cmd_plot(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        if not rt.session:
            await update.effective_message.reply_text("⚠️ Please load a character first using /char.")
            return
        text = " ".join(context.args).strip()
        if not text:
            cur = rt.session.plot_note or "(Not set)"
            await update.effective_message.reply_text(f"📌 Current plot direction (@Depth 2):\n{cur}")
            return
        rt.session.plot_note = text
        self.storage.save_session(rt.session)
        await update.effective_message.reply_text(f"📌 Plot direction updated (@Depth 2):\n{text}")

    async def cmd_plot_once(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        if not rt.session:
            await update.effective_message.reply_text("⚠️ Please load a character first using /char.")
            return
        text = " ".join(context.args).strip()
        if not text:
            cur = rt.session.plot_once or "(Not set)"
            await update.effective_message.reply_text(f"⚡ Current single-turn plot event:\n{cur}")
            return
        rt.session.plot_once = text
        self.storage.save_session(rt.session)
        await update.effective_message.reply_text(f"⚡ Single-turn plot event set (applies to the next reply only):\n{text}")

    async def cmd_plot_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        if not rt.session:
            await update.effective_message.reply_text("⚠️ Please load a character first using /char.")
            return
        rt.session.plot_note = ""
        rt.session.plot_once = ""
        self.storage.save_session(rt.session)
        await update.effective_message.reply_text("🧹 Cleared the plot direction and single-turn event.")

    async def cmd_persona(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        if not rt.session:
            await update.effective_message.reply_text("⚠️ Please load a character first using /char.")
            return
        text = " ".join(context.args).strip()
        if not text:
            cur = rt.session.persona_override or "(not overridden; using the character card defaults)"
            await update.effective_message.reply_text(f"🎭 Current runtime persona override (@Depth 4):\n{cur}")
            return
        rt.session.persona_override = text
        self.storage.save_session(rt.session)
        await update.effective_message.reply_text(f"🎭 Runtime persona override updated (@Depth 4):\n{text}")

    async def cmd_style(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        if not rt.session:
            await update.effective_message.reply_text("⚠️ Please load a character first using /char.")
            return
        text = " ".join(context.args).strip()
        if not text:
            cur = rt.session.style_override or "(not overridden)"
            await update.effective_message.reply_text(f"✍️ Current tone and formatting override (@Depth 4):\n{cur}")
            return
        rt.session.style_override = text
        self.storage.save_session(rt.session)
        await update.effective_message.reply_text(f"✍️ Output tone and formatting override updated (@Depth 4):\n{text}")

    async def cmd_reset_trait(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        if not rt.session:
            await update.effective_message.reply_text("⚠️ Please load a character first using /char.")
            return
        rt.session.persona_override = ""
        rt.session.style_override = ""
        self.storage.save_session(rt.session)
        await update.effective_message.reply_text("♻️ Cleared all persona and style overrides; character card defaults restored.")

    async def cmd_ooc(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        if not rt.session or not rt.character:
            await update.effective_message.reply_text("⚠️ Please load a character first using /char.")
            return
        question = " ".join(context.args).strip()
        if not question:
            await update.effective_message.reply_text("Usage: /ooc <question to discuss out of character>")
            return
        assembled = await self.engine.build_prompt(
            session=rt.session,
            character=rt.character,
            model_key=self.kobold.active_model_key,
            ooc_question=question,
        )
        await self._run_generation(
            rt=rt,
            bot=context.bot,
            chat_id=update.effective_chat.id,
            target_message=update.effective_message,
            assembled=assembled,
            is_ooc=True,
        )

    async def cmd_dir(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        if not rt.session or not rt.character:
            await update.effective_message.reply_text("⚠️ Please load a character first using /char.")
            return
        instruction = " ".join(context.args).strip()
        if not instruction:
            await update.effective_message.reply_text("Usage: /dir <director instruction, e.g. make her notice it is raining outside>")
            return
        if not rt.session.turns or rt.session.turns[-1].role == "assistant":
            rt.session.append_turn(
                role="user",
                text=f"*(🎬 Scene pushed: {instruction})*",
                telegram_message_id=update.effective_message.message_id,
            )
            self.storage.save_session(rt.session)
        assembled = await self.engine.build_prompt(
            session=rt.session,
            character=rt.character,
            model_key=self.kobold.active_model_key,
            inline_ooc=f"Director instruction: {instruction}",
        )
        await self._run_generation(
            rt=rt,
            bot=context.bot,
            chat_id=update.effective_chat.id,
            target_message=update.effective_message,
            assembled=assembled,
        )

    async def cmd_retry(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        if not rt.session or not rt.character:
            await update.effective_message.reply_text("⚠️ Please load a character first using /char.")
            return
        guidance = " ".join(context.args).strip() or None
        await self._perform_retry(
            rt=rt,
            bot=context.bot,
            chat_id=update.effective_chat.id,
            fallback_message=update.effective_message,
            guidance=guidance,
        )

    async def _perform_retry(
        self,
        rt: ChatRuntime,
        bot: Any,
        chat_id: int,
        fallback_message: Any,
        guidance: Optional[str] = None,
        target_turn: Optional[Turn] = None,
    ) -> None:
        await self.kobold.abort_generation()
        assert rt.session is not None and rt.character is not None

        last_turn = target_turn or rt.session.get_last_turn()
        if last_turn is None:
            await fallback_message.reply_text("⚠️ No reply available to reroll.")
            return

        # Build prompt up to right before target_turn if target_turn is an assistant message
        original_turns = rt.session.turns
        if last_turn.role == "assistant":
            try:
                t_idx = original_turns.index(last_turn)
                prompt_session = SessionState(**{
                    **rt.session.__dict__,
                    "turns": original_turns[:t_idx],
                })
            except ValueError:
                prompt_session = rt.session
            exclude_last = False
        else:
            prompt_session = rt.session
            exclude_last = False

        assembled = await self.engine.build_prompt(
            session=prompt_session,
            character=rt.character,
            model_key=self.kobold.active_model_key,
            rewrite_guidance=guidance,
            exclude_last_assistant=exclude_last,
        )

        seed = random.randint(1, 2_147_483_647)
        model_info = self.kobold.get_model_info(self.kobold.active_model_key)
        base_temp = float((model_info.get("sampling") or {}).get("temp", 0.9))
        temp_override = min(1.2, base_temp + 0.05) if not guidance else base_temp

        edit_msg_id = last_turn.telegram_message_id if last_turn.role == "assistant" else None
        await self._run_generation(
            rt=rt,
            bot=bot,
            chat_id=chat_id,
            target_message=fallback_message,
            assembled=assembled,
            edit_in_place=bool(edit_msg_id),
            target_msg_id=edit_msg_id,
            seed=seed,
            temp_override=temp_override,
            target_turn=last_turn if last_turn.role == "assistant" else None,
        )

    async def _perform_continue(
        self,
        rt: ChatRuntime,
        bot: Any,
        chat_id: int,
        fallback_message: Any,
        target_turn: Optional[Turn] = None,
    ) -> None:
        await self.kobold.abort_generation()
        assert rt.session is not None and rt.character is not None

        last_turn = target_turn or rt.session.get_last_turn()
        if last_turn is None or last_turn.role != "assistant":
            await fallback_message.reply_text("⚠️ Continuation is only supported for character replies.")
            return

        original_turns = rt.session.turns
        try:
            t_idx = original_turns.index(last_turn)
            prompt_session = SessionState(**{
                **rt.session.__dict__,
                "turns": original_turns[: t_idx + 1],
            })
        except ValueError:
            prompt_session = rt.session

        assembled = await self.engine.build_prompt(
            session=prompt_session,
            character=rt.character,
            model_key=self.kobold.active_model_key,
            continue_mode=True,
        )

        edit_msg_id = last_turn.telegram_message_id or (
            last_turn.get_message_ids()[0] if last_turn.get_message_ids() else None
        )
        await self._run_generation(
            rt=rt,
            bot=bot,
            chat_id=chat_id,
            target_message=fallback_message,
            assembled=assembled,
            edit_in_place=bool(edit_msg_id),
            target_msg_id=edit_msg_id,
            target_turn=last_turn,
            continue_append=True,
        )

    async def cmd_edit(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        if not rt.session or not rt.character:
            await update.effective_message.reply_text("⚠️ Please load a character first using /char.")
            return
        raw_new = " ".join(context.args).strip()
        if not raw_new:
            await update.effective_message.reply_text("Usage: /edit <corrected user line>")
            return

        clean_dialogue, inline_ooc = self.engine.parse_user_input(raw_new)
        last_user = rt.session.get_last_turn(role="user")
        if last_user is None:
            await update.effective_message.reply_text("⚠️ No user turn available to edit.")
            return

        last_user.set_text(clean_dialogue or raw_new)
        idx = rt.session.turns.index(last_user)
        trailing = rt.session.turns[idx + 1 :]
        target_assistant: Optional[Turn] = None
        if len(trailing) == 1 and trailing[0].role == "assistant":
            target_assistant = trailing[0]
        elif trailing:
            rt.session.turns = rt.session.turns[: idx + 1]

        self.storage.save_session(rt.session)

        assembled = await self.engine.build_prompt(
            session=rt.session,
            character=rt.character,
            model_key=self.kobold.active_model_key,
            inline_ooc=inline_ooc,
            exclude_last_assistant=(target_assistant is not None),
        )
        edit_msg_id = target_assistant.telegram_message_id if target_assistant else None
        await self._run_generation(
            rt=rt,
            bot=context.bot,
            chat_id=update.effective_chat.id,
            target_message=update.effective_message,
            assembled=assembled,
            edit_in_place=bool(edit_msg_id),
            target_msg_id=edit_msg_id,
            target_turn=target_assistant,
        )

    async def cmd_rewind(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        if not rt.session:
            await update.effective_message.reply_text("⚠️ Please load a character first using /char.")
            return
        n = 1
        if context.args:
            try:
                n = max(1, int(context.args[0]))
            except ValueError:
                await update.effective_message.reply_text("Usage: /rewind <turns to rewind, default 1>")
                return

        if len(rt.session.turns) <= 1:
            await update.effective_message.reply_text("⚠️ Already at the start of the story; cannot rewind further.")
            return

        cp_path = self.storage.create_checkpoint(rt.session, custom_title="rewind_backup")

        removed = 0
        user_turns_removed = 0
        while len(rt.session.turns) > 1 and user_turns_removed < n:
            popped = rt.session.turns.pop()
            removed += 1
            if popped.role == "user":
                user_turns_removed += 1

        rt.awaiting_rewrite_guidance = False
        rt.rewrite_target_turn_id = None
        last_turn = rt.session.get_last_turn()
        if last_turn and rt.session.summary_upto_turn_id >= last_turn.turn_id:
            rt.session.summary_upto_turn_id = max(0, last_turn.turn_id - 4)

        self.storage.save_session(rt.session)
        last_turn = rt.session.get_last_turn()
        preview = last_turn.text[:160] + ("…" if len(last_turn.text) > 160 else "") if last_turn else "(none)"
        await self._safe_send_html(
            update.effective_message,
            f"⏪ Rewound {removed} messages (auto-backup branch: <code>{_esc(cp_path.name)}</code>).\n\n"
            f"📍 Latest conversation excerpt:\n{_esc(preview)}",
            already_html=True,
        )

    async def cmd_save(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        if not rt.session:
            await update.effective_message.reply_text("⚠️ No active story to save.")
            return
        title = " ".join(context.args).strip()
        cp_path = self.storage.create_checkpoint(rt.session, custom_title=title)
        md_path = self.storage.export_story_markdown(rt.session, rt.character, custom_title=title)
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🧹 Unload Model Memory (Unload)", callback_data="unload_model")]
        ])
        await self._safe_send_html(
            update.effective_message,
            f"💾 Saved and exported!\n• Checkpoint: <code>{_esc(cp_path.name)}</code>\n• Novel file: <code>{_esc(md_path.name)}</code>",
            reply_markup=kb,
            already_html=True,
        )
        with open(md_path, "rb") as f:
            await update.effective_message.reply_document(document=f, filename=md_path.name)

    @staticmethod
    def _format_unload_summary(summary: dict[str, list[Any]]) -> str:
        parts: list[str] = []
        if summary.get("kobold_stopped"):
            parts.append(f"Stopped KoboldCpp (PID: {', '.join(str(p) for p in summary['kobold_stopped'])})")
        if summary.get("ollama_unloaded"):
            parts.append(f"Unloaded Ollama models ({', '.join(summary['ollama_unloaded'])})")
        if not parts:
            return "ℹ️ No model loaded; memory is already free."
        return "🧹 Freed model memory:\n• " + "\n• ".join(parts)

    async def cmd_exit(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        if rt.generating and rt.cancel_event:
            rt.cancel_event.set()
            await self.kobold.abort_generation()
        if rt.session:
            self.storage.save_session(rt.session)
        summary = await self.kobold.unload_all_models()
        msg = self._format_unload_summary(summary)
        if rt.session:
            msg = f"💾 Progress auto-saved.\n{msg}"
        await update.effective_message.reply_text(msg)

    async def cmd_load(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        sessions = self.storage.list_sessions(limit=15, chat_id=update.effective_chat.id)
        if not sessions:
            await update.effective_message.reply_text("⚠️ No saved sessions found in sessions/.")
            return
        rows: list[list[InlineKeyboardButton]] = []
        for s in sessions:
            sid = rt.register_id("ses", str(s["path"]))
            tag = "🔖" if s["is_checkpoint"] else "💬"
            ts = s["updated_at"][5:16].replace("T", " ")
            label = f"{tag} {s['char_name']} | {ts} ({s['turn_count']} turns)"
            rows.append([InlineKeyboardButton(label[:60], callback_data=f"load_ses:{sid}")])
        await update.effective_message.reply_text(
            "📂 Choose a saved session or checkpoint to load:",
            reply_markup=InlineKeyboardMarkup(rows),
        )

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        m_key = self.kobold.active_model_key
        m_info = self.kobold.models.get(m_key, {})
        live_model = await self.kobold.get_loaded_model_from_api()
        live_str = f"🟢 Running (<code>{_esc(live_model)}</code>)" if live_model else "🔴 Not running / offline"

        if rt.session and rt.character:
            assembled = await self.engine.build_prompt(
                session=rt.session,
                character=rt.character,
                model_key=m_key,
            )
            ctx_usage = f"{assembled.prompt_tokens:,} / {assembled.max_context:,} tokens"
            lore_list = ", ".join(rt.session.bound_lorebooks) if rt.session.bound_lorebooks else "none"
            temp_cnt = len(rt.session.temp_lore_entries)
            trig = ", ".join(assembled.triggered_lore_keys) if assembled.triggered_lore_keys else "none"
            plot_str = rt.session.plot_note or "(Not set)"
            plot_once_str = rt.session.plot_once or "(Not set)"
            persona_str = rt.session.persona_override or "(Default)"
            style_str = rt.session.style_override or "(Default)"
            char_str = f"{rt.character.name} ({len(rt.session.turns)} turns)"
            summary_str = (
                f"covers through Turn #{rt.session.summary_upto_turn_id} ({len(rt.session.story_summary)} chars)"
                if rt.session.story_summary
                else "(Not created yet)"
            )
            arc = rt.session.story_arc or {}
            if arc and (arc.get("title") or arc.get("chapters")):
                chs = arc.get("chapters") or []
                cur_ch = min(int(arc.get("current_chapter", 0)) + 1, max(1, len(chs)))
                state_txt = "Active" if arc.get("enabled", True) else "Paused"
                arc_str = f"\"{arc.get('title', 'Untitled')}\" [Chapter {cur_ch}/{max(1, len(chs))} · {state_txt}]"
            else:
                arc_str = "(Not set)"
        else:
            ctx_usage = f"0 / {m_info.get('context_size', 65536):,} tokens"
            lore_list = "none"
            temp_cnt = 0
            trig = "none"
            plot_str = plot_once_str = persona_str = style_str = "(No character loaded)"
            char_str = "Not loaded"
            summary_str = "(No character loaded)"
            arc_str = "(No character loaded)"

        msg = (
            "📊 <b>System Status Dashboard</b>\n\n"
            f"• <b>Selected Model</b>: {_esc(m_info.get('display_name', m_key))} (<code>{_esc(m_info.get('instruct_template', 'chatml'))}</code>)\n"
            f"• <b>Inference Node</b>: {live_str}\n"
            f"• <b>Context Usage</b>: {_esc(ctx_usage)}\n"
            f"• <b>Character</b>: {_esc(char_str)}\n"
            f"• <b>Story Arc (/arc)</b>: {_esc(arc_str)}\n"
            f"• <b>Story Summary (/summarize)</b>: {_esc(summary_str)}\n"
            f"• <b>Lorebooks</b>: {_esc(lore_list)} (temp entries: {temp_cnt})\n"
            f"• <b>Triggered Lore</b>: {_esc(trig)}\n"
            f"• <b>Plot Note (/plot)</b>: {_esc(plot_str)}\n"
            f"• <b>Next-Turn Event (/plot_once)</b>: {_esc(plot_once_str)}\n"
            f"• <b>Persona Override (/persona)</b>: {_esc(persona_str)}\n"
            f"• <b>Style Override (/style)</b>: {_esc(style_str)}"
        )
        await self._safe_send_html(update.effective_message, msg, already_html=True)

    # ------------------------------------------------------------------
    # Message & Callback Handlers
    # ------------------------------------------------------------------
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        if not update.effective_message or not update.effective_message.text:
            return
        rt = self.get_runtime(update.effective_chat.id)
        raw_text = update.effective_message.text.strip()
        if not raw_text:
            return

        if rt.awaiting_rewrite_guidance:
            rt.awaiting_rewrite_guidance = False
            tid = rt.rewrite_target_turn_id
            rt.rewrite_target_turn_id = None
            if raw_text.lower() in {"cancel", "cancel", "/cancel"}:
                await update.effective_message.reply_text("ℹ️ Rewrite guidance cancelled.")
                return
            target_turn = None
            if rt.session and tid is not None:
                for t in rt.session.turns:
                    if t.turn_id == tid:
                        target_turn = t
                        break
            await self._perform_retry(
                rt=rt,
                bot=context.bot,
                chat_id=update.effective_chat.id,
                fallback_message=update.effective_message,
                guidance=raw_text,
                target_turn=target_turn,
            )
            return

        if rt.awaiting_arc_theme:
            rt.awaiting_arc_theme = False
            pending_char_fn = rt.pending_arc_char_filename
            if raw_text.lower() in {"cancel", "cancel", "/cancel"}:
                rt.pending_arc_char_filename = None
                await update.effective_message.reply_text("ℹ️ Custom story arc theme cancelled.")
                return
            if pending_char_fn:
                rt.pending_arc_char_filename = None
                char = self.storage.load_character(pending_char_fn)
                user_name = str(self.bot_config.get("default_user_name", "Player"))
                status_msg = await update.effective_message.reply_text(
                    f"🗺️ Weaving a story arc from \"{raw_text}\" and {char.name}'s lorebooks..."
                )
                try:
                    arc = await self._generate_story_arc_json(
                        character=char, user_name=user_name, theme_hint=raw_text
                    )
                    rt.pending_arc_data = arc
                    nsid = rt.register_id("chr_act", pending_char_fn)
                    preview_text, _ = self._format_arc_display(arc)
                    kb = InlineKeyboardMarkup([
                        [InlineKeyboardButton("▶️ Generate Custom Opening & Start", callback_data=f"arc_start:custom:{nsid}")],
                        [InlineKeyboardButton("📖 Use Original Opening + Story Arc", callback_data=f"arc_start:orig:{nsid}")],
                        [InlineKeyboardButton("🔄 Regenerate Outline", callback_data=f"arc_theme:auto:{nsid}")],
                    ])
                    await self._safe_edit_text(
                        context.bot,
                        update.effective_chat.id,
                        status_msg.message_id,
                        preview_text,
                        reply_markup=kb,
                        already_html=True,
                    )
                except Exception as exc:
                    await status_msg.edit_text(f"❌ Failed to generate the story arc: {exc}")
                return
            if rt.session and rt.character:
                status_msg = await update.effective_message.reply_text("🗺️ Generating the story arc...")
                try:
                    arc = await self._generate_story_arc_json(
                        character=rt.character,
                        user_name=rt.session.user_name,
                        theme_hint=raw_text,
                        session=rt.session,
                    )
                    rt.session.story_arc = arc
                    self.storage.save_session(rt.session)
                    text, kb = self._format_arc_display(arc)
                    await self._safe_edit_text(
                        context.bot,
                        update.effective_chat.id,
                        status_msg.message_id,
                        text,
                        reply_markup=kb,
                        already_html=True,
                    )
                except Exception as exc:
                    await status_msg.edit_text(f"❌ Failed to generate the story arc: {exc}")
                return

        if not rt.session or not rt.character:
            await update.effective_message.reply_text("⚠️ Please select a character first using /char.")
            return

        clean_dialogue, inline_ooc = self.engine.parse_user_input(raw_text)
        if not clean_dialogue and inline_ooc:
            last_turn = rt.session.get_last_turn()
            if last_turn is not None and last_turn.role == "assistant":
                await self._perform_retry(
                    rt=rt,
                    bot=context.bot,
                    chat_id=update.effective_chat.id,
                    fallback_message=update.effective_message,
                    guidance=inline_ooc,
                    target_turn=last_turn,
                )
                return
            rt.session.append_turn(
                role="user",
                text="[Scene push]",
                telegram_message_id=update.effective_message.message_id,
            )
            self.storage.save_session(rt.session)
        elif clean_dialogue:
            rt.session.append_turn(
                role="user",
                text=clean_dialogue,
                telegram_message_id=update.effective_message.message_id,
            )
            self.storage.save_session(rt.session)

        assembled = await self.engine.build_prompt(
            session=rt.session,
            character=rt.character,
            model_key=self.kobold.active_model_key,
            inline_ooc=inline_ooc,
        )

        await self._run_generation(
            rt=rt,
            bot=context.bot,
            chat_id=update.effective_chat.id,
            target_message=update.effective_message,
            assembled=assembled,
        )

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        query = update.callback_query
        if not query or not query.data:
            return
        await query.answer()
        data = query.data
        chat_id = update.effective_chat.id
        rt = self.get_runtime(chat_id)
        if not data.startswith("sw:guide:"):
            rt.awaiting_rewrite_guidance = False
            rt.rewrite_target_turn_id = None
        if not data.startswith(("chr_arc:", "arc_theme:", "arc_ctl:")):
            rt.awaiting_arc_theme = False

        if data == "stop_gen":
            if rt.cancel_event:
                rt.cancel_event.set()
            await self.kobold.abort_generation()
            return

        if data == "unload_model":
            if rt.session:
                self.storage.save_session(rt.session)
            summary = await self.kobold.unload_all_models()
            await query.edit_message_text(self._format_unload_summary(summary))
            return

        if data == "mdl_fw_back":
            live_model = await self.kobold.get_loaded_model_from_api()
            text, markup = self._build_model_framework_menu(rt, live_model)
            await query.edit_message_text(text, reply_markup=markup)
            return

        if data.startswith("mdl_fw:"):
            backend = data.split(":", 1)[1].lower()
            live_model = await self.kobold.get_loaded_model_from_api()
            text, markup = self._build_model_list_menu(rt, backend, live_model)
            await query.edit_message_text(text, reply_markup=markup)
            return

        if data.startswith("mdl:"):
            sid = data.split(":", 1)[1]
            model_key = rt.resolve_id("mdl", sid)
            if not model_key or model_key not in self.kobold.models:
                await query.edit_message_text("⚠️ Model options expired. Run /model again.")
                return
            info = self.kobold.get_model_info(model_key)
            display_name = info.get("display_name", model_key)
            tpl_name = info.get("instruct_template", "chatml")
            await self._safe_edit_text(
                context.bot,
                chat_id,
                query.message.message_id,
                f"⏳ Unloading the previous model and loading <code>{_esc(display_name)}</code>...",
                already_html=True,
            )
            if rt.session:
                self.storage.save_session(rt.session)
            try:
                async def _progress(msg: str) -> None:
                    await self._safe_edit_text(context.bot, chat_id, query.message.message_id, msg, use_html=False)

                loaded_name = await self.kobold.switch_model(model_key, progress_callback=_progress)
                if rt.session:
                    rt.session.active_model = model_key
                    self.storage.save_session(rt.session)
                await self._safe_edit_text(
                    context.bot,
                    chat_id,
                    query.message.message_id,
                    f"✅ Model <b>{_esc(display_name)}</b> loaded!\n"
                    f"• Reported name: <code>{_esc(loaded_name)}</code>\n"
                    f"• Instruct template set to: <code>{_esc(tpl_name)}</code>",
                    already_html=True,
                )
            except Exception as exc:
                await query.edit_message_text(f"❌ Failed to switch models: {exc}")
            return

        if data.startswith("chr_pick:"):
            sid = data.split(":", 1)[1]
            filename = rt.resolve_id("chr", sid)
            if not filename:
                await query.edit_message_text("⚠️ Character options expired. Run /char again.")
                return
            char = self.storage.load_character(filename)
            nsid = rt.register_id("chr_act", filename)
            lb_text = ", ".join(char.bound_lorebooks) if char.bound_lorebooks else "none"
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("📖 New Story (First Message)", callback_data=f"chr_new:{nsid}")],
                [InlineKeyboardButton("🗺️ Generate World Story Arc (Campaign)", callback_data=f"chr_arc:{nsid}")],
                [InlineKeyboardButton("📂 Resume Latest Session", callback_data=f"chr_res:{nsid}")],
            ])
            await self._safe_edit_text(
                context.bot,
                chat_id,
                query.message.message_id,
                f"🎭 Character selected: <b>{_esc(char.name)}</b>\n"
                f"• Default lorebooks: {_esc(lb_text)}\n"
                "Choose to start a new story or resume progress:",
                reply_markup=kb,
                already_html=True,
            )
            return

        if data.startswith("chr_arc:"):
            sid = data.split(":", 1)[1]
            filename = rt.resolve_id("chr_act", sid)
            if not filename:
                await query.edit_message_text("⚠️ Character options expired. Run /char again.")
                return
            char = self.storage.load_character(filename)
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("🎲 Freeform World-Building", callback_data=f"arc_theme:auto:{sid}")],
                [InlineKeyboardButton("🔍 Mystery & Investigation", callback_data=f"arc_theme:mystery:{sid}")],
                [InlineKeyboardButton("💕 Slice of Life & Romance", callback_data=f"arc_theme:romance:{sid}")],
                [InlineKeyboardButton("⚔️ Crisis & Shared Trial", callback_data=f"arc_theme:crisis:{sid}")],
                [InlineKeyboardButton("✏️ Enter Custom Theme", callback_data=f"arc_theme:custom:{sid}")],
            ])
            await self._safe_edit_text(
                context.bot,
                chat_id,
                query.message.message_id,
                f"🗺️ <b>Design a story arc for {_esc(char.name)}</b>\nPick the tone you want (you can always diverge from the arc during play):",
                reply_markup=kb,
                already_html=True,
            )
            return

        if data.startswith("arc_theme:"):
            parts = data.split(":", 2)
            if len(parts) < 3:
                return
            _, mode, sid = parts
            filename = rt.resolve_id("chr_act", sid)
            if not filename:
                await query.edit_message_text("⚠️ Character options expired. Run /char again.")
                return
            char = self.storage.load_character(filename)
            if mode == "custom":
                rt.awaiting_arc_theme = True
                rt.pending_arc_char_filename = filename
                await query.edit_message_text(
                    f"✏️ Send the arc theme or opening premise for \"{char.name}\" (type \"cancel\" to abort):"
                )
                return
            theme_map = {
                "auto": "",
                "mystery": "Mystery investigation and adventure puzzles built on lore locations and secrets",
                "romance": "Grow closer and warm the relationship through daily life and events",
                "crisis": "Sudden crisis and a shared trial both must overcome together",
            }
            theme_hint = theme_map.get(mode, "")
            await self._safe_edit_text(
                context.bot,
                chat_id,
                query.message.message_id,
                f"🗺️ Drafting a 4-chapter arc from <b>{_esc(char.name)}</b> and the lorebooks...",
                already_html=True,
            )
            user_name = str(self.bot_config.get("default_user_name", "Player"))
            try:
                arc = await self._generate_story_arc_json(
                    character=char, user_name=user_name, theme_hint=theme_hint
                )
                rt.pending_arc_data = arc
                rt.pending_arc_char_filename = filename
                preview_text, _ = self._format_arc_display(arc)
                kb = InlineKeyboardMarkup([
                    [InlineKeyboardButton("▶️ Generate Custom Opening & Start", callback_data=f"arc_start:custom:{sid}")],
                    [InlineKeyboardButton("📖 Use Original Opening + Story Arc", callback_data=f"arc_start:orig:{sid}")],
                    [InlineKeyboardButton("🔄 Regenerate Outline", callback_data=f"arc_theme:{mode}:{sid}")],
                ])
                await self._safe_edit_text(
                    context.bot,
                    chat_id,
                    query.message.message_id,
                    preview_text,
                    reply_markup=kb,
                    already_html=True,
                )
            except Exception as exc:
                await query.edit_message_text(f"❌ Failed to generate the story arc: {exc}")
            return

        if data.startswith("arc_start:"):
            parts = data.split(":", 2)
            if len(parts) < 3:
                return
            _, kind, sid = parts
            filename = rt.resolve_id("chr_act", sid)
            if not filename:
                await query.edit_message_text("⚠️ Character options expired. Run /char again.")
                return
            char = self.storage.load_character(filename)
            arc = rt.pending_arc_data or {}
            rt.pending_arc_data = None
            rt.pending_arc_char_filename = None
            user_name = str(self.bot_config.get("default_user_name", "Player"))
            sess = self.storage.create_session(
                character=char,
                active_model=self.kobold.active_model_key,
                user_name=user_name,
                include_first_mes=(kind != "custom"),
                chat_id=chat_id,
            )
            sess.story_arc = json.loads(json.dumps(arc)) if arc else {}
            rt.character = char
            rt.session = sess
            lb_str = ", ".join(sess.bound_lorebooks) if sess.bound_lorebooks else "none"
            if kind == "custom" and arc:
                await self._safe_edit_text(
                    context.bot,
                    chat_id,
                    query.message.message_id,
                    f"🗺️ Mounted the arc \"<b>{_esc(arc.get('title', char.name))}</b>\" and writing the prologue...",
                    already_html=True,
                )
                try:
                    opening = await self._generate_arc_opening_scene(char, user_name, arc, sess)
                except Exception:
                    opening = self.engine.replace_macros(char.first_mes, char.name, user_name)
                if opening:
                    sess.append_turn(role="assistant", text=opening)
            else:
                await self._safe_edit_text(
                    context.bot,
                    chat_id,
                    query.message.message_id,
                    f"📖 Started the arc story with <b>{_esc(char.name)}</b>!\n"
                    f"• Arc: {_esc(arc.get('title', 'Untitled'))}\n"
                    f"• Auto-mounted lorebooks: {_esc(lb_str)}",
                    already_html=True,
                )
            if sess.turns:
                turn0 = sess.turns[-1]
                turn0.swipes = [
                    self.engine.replace_macros(s, char.name, sess.user_name)
                    for s in turn0.swipes
                ]
                sent = await self._safe_send_html(query.message, turn0.text, reply_markup=self.swipe_keyboard(turn0))
                if sent:
                    turn0.set_message_ids(self._last_chunk_ids or [sent.message_id])
            self.storage.save_session(sess)
            return

        if data.startswith("arc_ctl:"):
            if not rt.session or not rt.character:
                await query.edit_message_text("⚠️ Load a character conversation first.")
                return
            action = data.split(":", 1)[1]
            sess = rt.session
            arc = dict(sess.story_arc or {})
            chapters = list(arc.get("chapters") or [])
            if action == "custom_prompt":
                rt.awaiting_arc_theme = True
                rt.pending_arc_char_filename = None
                await query.edit_message_text("✏️ Send the arc theme for the current story (type \"cancel\" to abort):")
                return
            if action == "prev" and chapters:
                cur = int(arc.get("current_chapter", 0))
                arc["current_chapter"] = max(0, cur - 1)
                sess.story_arc = arc
                self.storage.save_session(sess)
                text, kb = self._format_arc_display(arc)
                await self._safe_edit_text(
                    context.bot, chat_id, query.message.message_id, text, reply_markup=kb, already_html=True
                )
                return
            if action == "next" and chapters:
                cur = int(arc.get("current_chapter", 0))
                arc["current_chapter"] = min(len(chapters) - 1, cur + 1)
                sess.story_arc = arc
                self.storage.save_session(sess)
                text, kb = self._format_arc_display(arc)
                await self._safe_edit_text(
                    context.bot, chat_id, query.message.message_id, text, reply_markup=kb, already_html=True
                )
                return
            if action == "toggle" and chapters:
                arc["enabled"] = not bool(arc.get("enabled", True))
                sess.story_arc = arc
                self.storage.save_session(sess)
                text, kb = self._format_arc_display(arc)
                await self._safe_edit_text(
                    context.bot, chat_id, query.message.message_id, text, reply_markup=kb, already_html=True
                )
                return
            if action == "play" and chapters:
                arc["enabled"] = True
                sess.story_arc = arc
                if len(sess.turns) <= 1:
                    await self._safe_edit_text(
                        context.bot,
                        chat_id,
                        query.message.message_id,
                        f"▶️ Starting arc \"<b>{_esc(arc.get('title', rt.character.name))}</b>\"...",
                        already_html=True,
                    )
                    try:
                        opening = await self._generate_arc_opening_scene(rt.character, sess.user_name, arc, sess)
                    except Exception:
                        opening = self.engine.replace_macros(rt.character.first_mes, rt.character.name, sess.user_name)
                    if opening:
                        turn = sess.append_turn(role="assistant", text=opening)
                        sent = await self._safe_send_html(query.message, turn.text, reply_markup=self.swipe_keyboard(turn))
                        if sent:
                            turn.set_message_ids(self._last_chunk_ids or [sent.message_id])
                    self.storage.save_session(sess)
                    return
                cur_idx = min(max(0, int(arc.get("current_chapter", 0))), len(chapters) - 1)
                ch = chapters[cur_idx] if isinstance(chapters[cur_idx], dict) else {}
                ch_name = str(ch.get("name", f"Chapter {cur_idx + 1}"))
                ch_goal = str(ch.get("goal", ""))
                await self._safe_edit_text(
                    context.bot,
                    chat_id,
                    query.message.message_id,
                    f"▶️ Continuing arc \"<b>{_esc(arc.get('title', rt.character.name))}</b>\" ({_esc(ch_name)})...",
                    already_html=True,
                )
                sess.plot_once = (
                    f"[Director: Naturally weave in the current chapter's beat "
                    f"({ch_name}: {ch_goal}) without resetting the scene or repeating earlier events.]"
                )
                self.storage.save_session(sess)
                await self._generate_turn(context.bot, chat_id, query.message, is_ooc=False)
                return
            if action == "adapt":
                await query.edit_message_text("🔀 Rebuilding the remaining chapters from the conversation...")
                try:
                    updated = await self._adapt_story_arc(rt)
                    text, kb = self._format_arc_display(updated)
                    await self._safe_edit_text(
                        context.bot, chat_id, query.message.message_id, text, reply_markup=kb, already_html=True
                    )
                except Exception as exc:
                    await query.edit_message_text(f"❌ Failed to rebuild the story arc: {exc}")
                return
            if action == "regen":
                await query.edit_message_text("🎲 Regenerating the story arc from character and lorebooks...")
                try:
                    new_arc = await self._generate_story_arc_json(
                        character=rt.character,
                        user_name=sess.user_name,
                        session=sess,
                    )
                    sess.story_arc = new_arc
                    self.storage.save_session(sess)
                    text, kb = self._format_arc_display(new_arc)
                    await self._safe_edit_text(
                        context.bot, chat_id, query.message.message_id, text, reply_markup=kb, already_html=True
                    )
                except Exception as exc:
                    await query.edit_message_text(f"❌ Regeneration failed: {exc}")
                return

        if data.startswith("chr_new:"):
            sid = data.split(":", 1)[1]
            filename = rt.resolve_id("chr_act", sid)
            if not filename:
                await query.edit_message_text("⚠️ Character options expired. Run /char again.")
                return
            char = self.storage.load_character(filename)
            user_name = str(self.bot_config.get("default_user_name", "Player"))
            sess = self.storage.create_session(
                character=char,
                active_model=self.kobold.active_model_key,
                user_name=user_name,
                include_first_mes=True,
                chat_id=chat_id,
            )
            rt.character = char
            rt.session = sess
            lb_str = ", ".join(sess.bound_lorebooks) if sess.bound_lorebooks else "none"
            await self._safe_edit_text(
                context.bot,
                chat_id,
                query.message.message_id,
                f"📖 Started a new story with <b>{_esc(char.name)}</b>!\n"
                f"• Auto-mounted lorebooks: {_esc(lb_str)}",
                already_html=True,
            )
            if sess.turns:
                turn0 = sess.turns[-1]
                turn0.swipes = [
                    self.engine.replace_macros(s, char.name, sess.user_name)
                    for s in turn0.swipes
                ]
                sent = await self._safe_send_html(query.message, turn0.text, reply_markup=self.swipe_keyboard(turn0))
                if sent:
                    turn0.set_message_ids(self._last_chunk_ids or [sent.message_id])
                self.storage.save_session(sess)
            return

        if data.startswith("chr_res:"):
            sid = data.split(":", 1)[1]
            filename = rt.resolve_id("chr_act", sid)
            if not filename:
                await query.edit_message_text("⚠️ Character options expired. Run /char again.")
                return
            char = self.storage.load_character(filename)
            sess = self.storage.get_latest_session_for_char(char.name, chat_id=chat_id)
            if sess is None:
                user_name = str(self.bot_config.get("default_user_name", "Player"))
                sess = self.storage.create_session(
                    character=char,
                    active_model=self.kobold.active_model_key,
                    user_name=user_name,
                    include_first_mes=True,
                    chat_id=chat_id,
                )
                rt.character = char
                rt.session = sess
                await query.edit_message_text(f"ℹ️ No previous save for {char.name}; started a new story.")
                if sess.turns:
                    turn0 = sess.turns[-1]
                    turn0.swipes = [
                        self.engine.replace_macros(s, char.name, sess.user_name)
                        for s in turn0.swipes
                    ]
                    sent = await self._safe_send_html(query.message, turn0.text, reply_markup=self.swipe_keyboard(turn0))
                    if sent:
                        turn0.set_message_ids(self._last_chunk_ids or [sent.message_id])
                    self.storage.save_session(sess)
                return

            rt.character = char
            rt.session = sess
            await self._send_session_recap(query, context.bot, chat_id, rt)
            return

        if data.startswith("world_toggle:"):
            if not rt.session:
                return
            sid = data.split(":", 1)[1]
            lb_name = rt.resolve_id("lb", sid)
            if not lb_name:
                return
            if lb_name in rt.session.bound_lorebooks:
                rt.session.bound_lorebooks.remove(lb_name)
            else:
                rt.session.bound_lorebooks.append(lb_name)
            self.storage.save_session(rt.session)
            await query.edit_message_reply_markup(reply_markup=self._world_keyboard(rt))
            return

        if data == "world_done":
            lb_str = ", ".join(rt.session.bound_lorebooks) if rt.session and rt.session.bound_lorebooks else "none"
            await query.edit_message_text(f"🌍 Lorebook mounts updated: {lb_str}")
            return

        if data.startswith("load_ses:"):
            sid = data.split(":", 1)[1]
            path_str = rt.resolve_id("ses", sid)
            if not path_str:
                await query.edit_message_text("⚠️ Save list expired. Run /load again.")
                return
            sess = self.storage.load_session(path_str)
            char = self.storage.load_character(sess.char_filename or sess.char_name)
            rt.session = sess
            rt.character = char
            await self._send_session_recap(query, context.bot, chat_id, rt)
            return

        if data.startswith("sw:"):
            if not rt.session or not rt.character:
                return
            _, action, tid_str = data.split(":", 2)
            tid = int(tid_str)
            target_turn = next((t for t in rt.session.turns if t.turn_id == tid), None)
            if not target_turn:
                return

            if action in ("prev", "next"):
                moved = target_turn.prev_swipe() if action == "prev" else target_turn.next_swipe()
                if moved:
                    old_ids = target_turn.get_message_ids()
                    first_id = old_ids[0] if old_ids else query.message.message_id
                    for extra_id in old_ids[1:]:
                        try:
                            await context.bot.delete_message(chat_id=chat_id, message_id=extra_id)
                        except Exception:
                            pass
                    kb = self.swipe_keyboard(target_turn)
                    if len(target_turn.text) <= 3800:
                        await self._safe_edit_text(
                            context.bot, chat_id, first_id, target_turn.text, reply_markup=kb, use_html=True
                        )
                        target_turn.set_message_ids([first_id])
                    else:
                        chunks = self._split_text(target_turn.text, 3800)
                        await self._safe_edit_text(
                            context.bot, chat_id, first_id, chunks[0], reply_markup=None, use_html=True
                        )
                        new_ids = [first_id]
                        for idx, chunk in enumerate(chunks[1:], start=1):
                            ckb = kb if idx == len(chunks) - 1 else None
                            sent_extra = await context.bot.send_message(
                                chat_id=chat_id,
                                text=self.engine.format_for_telegram_html(chunk),
                                parse_mode=ParseMode.HTML,
                                reply_markup=ckb,
                            )
                            new_ids.append(sent_extra.message_id)
                        target_turn.set_message_ids(new_ids)
                    self.storage.save_session(rt.session)
            elif action == "reroll":
                if not target_turn.get_message_ids():
                    target_turn.set_message_ids([query.message.message_id])
                await self._perform_retry(
                    rt=rt,
                    bot=context.bot,
                    chat_id=chat_id,
                    fallback_message=query.message,
                    target_turn=target_turn,
                )
            elif action == "guide":
                if not target_turn.get_message_ids():
                    target_turn.set_message_ids([query.message.message_id])
                rt.awaiting_rewrite_guidance = True
                rt.rewrite_target_turn_id = target_turn.turn_id
                await query.message.reply_text("✏️ Send a rewrite direction (the next message rewrites this reply in place):")
            elif action == "cont":
                if not target_turn.get_message_ids():
                    target_turn.set_message_ids([query.message.message_id])
                await self._perform_continue(
                    rt=rt,
                    bot=context.bot,
                    chat_id=chat_id,
                    fallback_message=query.message,
                    target_turn=target_turn,
                )
            elif action == "illustrate":
                await self._perform_illustrate(
                    rt=rt,
                    bot=context.bot,
                    chat_id=chat_id,
                    reply_to_message=query.message,
                    target_turn=target_turn,
                )

    async def _send_session_recap(self, query: Any, bot: Any, chat_id: int, rt: ChatRuntime) -> None:
        assert rt.session is not None and rt.character is not None
        sess = rt.session
        lb_str = ", ".join(sess.bound_lorebooks) if sess.bound_lorebooks else "none"
        await self._safe_edit_text(
            bot,
            chat_id,
            query.message.message_id,
            f"📂 Loaded: <b>{_esc(sess.char_name)}</b> (<code>{_esc(sess.file_path.name)}</code>)\n"
            f"• Lorebooks mounted: {_esc(lb_str)}\n"
            f"• Plot note: {_esc(sess.plot_note or 'none')}\n"
            f"• Total turns: {len(sess.turns)}\n\n"
            "Here is a recap of the last 2 turns:",
            already_html=True,
        )
        tail_turns = sess.turns[-2:]
        for idx, t in enumerate(tail_turns):
            is_final_assistant = idx == len(tail_turns) - 1 and t.role == "assistant"
            if is_final_assistant:
                sent = await self._safe_send_html(query.message, t.text, reply_markup=self.swipe_keyboard(t))
                if sent:
                    t.set_message_ids(self._last_chunk_ids or [sent.message_id])
                    self.storage.save_session(sess)
            else:
                speaker = sess.user_name if t.role == "user" else sess.char_name
                await self._safe_send_html(query.message, f"【{speaker}】\n{t.text}")

    async def check_idle_and_unload(self, bot: Any) -> bool:
        if not self.idle_unload_enabled or self.idle_unload_seconds <= 0:
            return False
        if any(rt.generating for rt in self.chats.values()):
            return False
        elapsed = time.monotonic() - self.kobold.last_activity_ts
        if elapsed < self.idle_unload_seconds:
            return False
        if not await self.kobold.has_any_loaded_model():
            return False

        for rt in self.chats.values():
            if rt.session:
                self.storage.save_session(rt.session)

        summary = await self.kobold.unload_all_models()
        self.kobold.touch_activity()
        did_unload = bool(summary.get("kobold_stopped") or summary.get("ollama_unloaded"))
        if did_unload:
            logger.info("Idle timeout (%ss) reached; unloaded models: %s", int(self.idle_unload_seconds), summary)
            if self.last_active_chat_id and bot is not None:
                try:
                    mins = max(1, int(self.idle_unload_seconds // 60))
                    await bot.send_message(
                        chat_id=self.last_active_chat_id,
                        text=f"💤 Idle for over {mins} minutes; model memory released automatically (the next message or /model reloads it).",
                    )
                except Exception as exc:
                    logger.debug("Failed to send idle unload notice: %s", exc)
        return did_unload

    async def idle_watchdog_loop(self, bot: Any, check_interval: float = 30.0) -> None:
        while True:
            await asyncio.sleep(check_interval)
            try:
                await self.check_idle_and_unload(bot)
            except Exception as exc:
                logger.warning("Idle watchdog error: %s", exc)

    # ------------------------------------------------------------------
    # Story Summary (/summarize & Auto-Trigger at 85% Context)
    # ------------------------------------------------------------------
    async def _summarize_oldest_turns(self, rt: ChatRuntime, count: int = 10) -> tuple[bool, str]:
        if not rt.session or not rt.character:
            return False, "No character or conversation loaded."
        sess = rt.session
        unsummarized = [t for t in sess.turns if t.turn_id > sess.summary_upto_turn_id]
        if len(unsummarized) <= 4:
            return False, "Not enough unsummarized history (the last 4 turns must be kept)."

        batch_size = min(count, len(unsummarized) - 4)
        batch = unsummarized[:batch_size]
        lines = []
        for t in batch:
            speaker = sess.user_name if t.role == "user" else sess.char_name
            lines.append(f"{speaker}: {t.text.strip()}")
        transcript = "\n\n".join(lines)

        prev_summary = sess.story_summary.strip()
        tpl_name = self.kobold.get_active_instruct_template()
        tpl = self.engine._get_template(tpl_name)
        prompt_body = (
            "Compress the roleplay transcript below into a recap of about 200 words."
            "Keep key plot turns, locations, relationship changes, and important promises. Add no commentary or formatting tags.\n\n"
        )
        if prev_summary:
            prompt_body += f"[Previous recap]\n{prev_summary}\n\n"
        prompt_body += f"[New transcript]\n{transcript}"

        assistant_start = str(
            tpl.get("assistant_prefill") or tpl.get("assistant_prefix", "<|im_start|>assistant\n")
        )
        full_prompt = (
            str(tpl.get("bos_token", ""))
            + self.engine._wrap_block("user", prompt_body, tpl)
            + assistant_start
        )
        model_info = self.kobold.get_model_info(self.kobold.active_model_key)
        sampling = dict(model_info.get("sampling") or {})
        sampling["max_length"] = 600

        chunks: list[str] = []
        async for tok in self.kobold.generate_stream(
            prompt=full_prompt,
            sampling_params=sampling,
            stop_sequences=list(tpl.get("stop_sequences") or ["<|im_end|>"]),
            max_context=int(model_info.get("context_size", 65536)),
            temp_override=0.4,
        ):
            chunks.append(tok)

        summary_text = self.engine.post_process_output("".join(chunks), sess.char_name, sess.user_name).strip()
        if not summary_text:
            return False, "The model did not produce a valid recap."

        sess.story_summary = summary_text
        sess.summary_upto_turn_id = batch[-1].turn_id
        self.storage.save_session(sess)
        if sess.story_arc and sess.story_arc.get("enabled", True) and sess.story_arc.get("chapters"):
            try:
                await self._adapt_story_arc(rt)
            except Exception as exc:
                logger.debug("Story arc post-summary sync skipped: %s", exc)
        return True, summary_text

    async def _auto_summarize_background(self, bot: Any, chat_id: int, rt: ChatRuntime) -> None:
        if rt.generating or chat_id in self._summarizing_chats:
            return
        self._summarizing_chats.add(chat_id)
        try:
            ok, result = await self._summarize_oldest_turns(rt, count=10)
            if ok:
                await bot.send_message(
                    chat_id=chat_id,
                    text=f"📝 Context hit the 85% cap; the oldest 10 turns were compressed into a recap:\n\n{result}",
                )
        except Exception as exc:
            logger.warning("Auto-summarize failed: %s", exc)
        finally:
            self._summarizing_chats.discard(chat_id)

    async def cmd_summarize(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        if not rt.session or not rt.character:
            await update.effective_message.reply_text("⚠️ No character or conversation loaded.")
            return
        status_msg = await update.effective_message.reply_text("📝 Compressing the oldest turns into a recap...")
        try:
            ok, result = await self._summarize_oldest_turns(rt, count=10)
            if ok:
                await status_msg.edit_text(f"✅ Recap updated (covers through Turn #{rt.session.summary_upto_turn_id}):\n\n{result}")
            else:
                await status_msg.edit_text(f"ℹ️ {result}")
        except Exception as exc:
            await status_msg.edit_text(f"❌ Failed to generate the recap: {exc}")

    # ------------------------------------------------------------------
    # Scene Illustration (/illustrate & ComfyUI Integration)
    # ------------------------------------------------------------------
    async def _generate_danbooru_tags(
        self, rt: ChatRuntime, target_turn: Optional[Turn] = None, extra_hint: str = ""
    ) -> tuple[str, str, list[tuple[str, float]], int]:
        assert rt.session is not None and rt.character is not None
        sess = rt.session
        if target_turn is not None:
            idx = next((i for i, t in enumerate(sess.turns) if t.turn_id == target_turn.turn_id), len(sess.turns) - 1)
            recent_turns = sess.turns[max(0, idx - 3) : idx + 1]
        else:
            recent_turns = sess.turns[-4:]

        scene_lines = []
        for t in recent_turns:
            speaker = sess.user_name if t.role == "user" else sess.char_name
            scene_lines.append(f"{speaker}: {t.text.strip()}")
        scene_text = "\n".join(scene_lines)

        persona_key = resolve_persona_for_character(rt.character.name)
        tpl_name = self.kobold.get_active_instruct_template()
        tpl = self.engine._get_template(tpl_name)
        if persona_key:
            prompt_body = (
                "Extract 15 to 28 concise, comma-separated English Danbooru tags describing ONLY the current scene "
                "(location/background, current clothing or undressed state, pose, action, facial expression, "
                "lighting, and camera angle) from the roleplay turns below.\n"
                "Do NOT invent or repeat character hair color, eye color, or character name (those are locked).\n"
                "Output ONLY comma-separated lowercase Danbooru tags, no sentences, no parentheses explanations, no markdown.\n\n"
                f"Recent scene:\n{scene_text}"
            )
        else:
            prompt_body = (
                "Convert the following anime roleplay scene into 20 to 35 concise, comma-separated English Danbooru tags "
                "for anime/Danbooru-style  image generation (character count, appearance, current outfit, pose, expression, "
                "background, lighting, camera angle). Output ONLY comma-separated tags, no explanations or markdown.\n\n"
                f"Character appearance reference: {rt.character.description[:400]}\n\n"
                f"Recent scene:\n{scene_text}"
            )
        if extra_hint:
            prompt_body += f"\n\nAdditional user hint: {extra_hint}"

        assistant_start = str(
            tpl.get("assistant_prefill") or tpl.get("assistant_prefix", "<|im_start|>assistant\n")
        )
        full_prompt = (
            str(tpl.get("bos_token", ""))
            + self.engine._wrap_block("user", prompt_body, tpl)
            + assistant_start
        )
        model_info = self.kobold.get_model_info(self.kobold.active_model_key)
        sampling = dict(model_info.get("sampling") or {})
        sampling["max_length"] = 140

        chunks: list[str] = []
        async for tok in self.kobold.generate_stream(
            prompt=full_prompt,
            sampling_params=sampling,
            stop_sequences=list(tpl.get("stop_sequences") or ["<|im_end|>"]),
            max_context=int(model_info.get("context_size", 65536)),
            temp_override=0.45,
        ):
            chunks.append(tok)

        tags = self.engine.post_process_output("".join(chunks), sess.char_name, sess.user_name)
        tags = re.sub(r"\([^)]*\)", "", tags)
        tags = re.sub(r"[\r\n]+", ", ", tags)
        tags = re.sub(r"\s*,\s*,+", ", ", tags).strip(" ,.")
        if not tags:
            tags = "indoor, warm lighting, upper body"

        return assemble_persona_prompt(persona_key, dynamic_tags=tags)

    async def _run_comfyui_txt2img(
        self,
        positive_tags: str,
        negative_tags: Optional[str] = None,
        loras: Optional[list[tuple[str, float]]] = None,
        clip_skip: int = 2,
    ) -> bytes:
        workflow = build_comfyui_workflow(
            positive=positive_tags,
            negative=negative_tags or DEFAULT_NEGATIVE,
            ckpt=self.comfyui_checkpoint,
            loras=loras,
            clip_skip=clip_skip,
            width=self.comfyui_width,
            height=self.comfyui_height,
            steps=self.comfyui_steps,
            cfg=self.comfyui_cfg,
            sampler=self.comfyui_sampler,
            scheduler=self.comfyui_scheduler,
            filename_prefix="stlite_illust",
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{self.comfyui_url}/prompt", json={"prompt": workflow})
            resp.raise_for_status()
            prompt_id = resp.json()["prompt_id"]

            for _ in range(90):
                await asyncio.sleep(1.5)
                hist_resp = await client.get(f"{self.comfyui_url}/history/{prompt_id}")
                if hist_resp.status_code != 200:
                    continue
                hist = hist_resp.json()
                if prompt_id not in hist:
                    continue
                outputs = hist[prompt_id].get("outputs", {})
                for node_out in outputs.values():
                    images = node_out.get("images") or []
                    if images:
                        img_meta = images[0]
                        view_resp = await client.get(
                            f"{self.comfyui_url}/view",
                            params={
                                "filename": img_meta["filename"],
                                "subfolder": img_meta.get("subfolder", ""),
                                "type": img_meta.get("type", "output"),
                            },
                        )
                        view_resp.raise_for_status()
                        img_bytes = view_resp.content
                        try:
                            await client.post(
                                f"{self.comfyui_url}/free",
                                json={"unload_models": True, "free_memory": True},
                            )
                        except Exception:
                            pass
                        return img_bytes
            raise TimeoutError("ComfyUI image generation timed out.")

    async def _perform_illustrate(
        self,
        rt: ChatRuntime,
        bot: Any,
        chat_id: int,
        reply_to_message: Any,
        target_turn: Optional[Turn] = None,
        extra_hint: str = "",
    ) -> None:
        if not rt.session or not rt.character:
            await reply_to_message.reply_text("⚠️ Load a character with /char and start a conversation first.")
            return
        status_msg = await reply_to_message.reply_text("🎨 Extracting scene tags and calling ComfyUI...")
        try:
            pos_tags, neg_tags, loras, clip_skip = await self._generate_danbooru_tags(
                rt, target_turn=target_turn, extra_hint=extra_hint
            )
            m_info = self.kobold.get_model_info(self.kobold.active_model_key)
            if float(m_info.get("vram_gb", 0) or 0) >= 18:
                await status_msg.edit_text(
                    f"🎨 Tags generated. Freeing model memory, then starting ComfyUI to draw...\n`{pos_tags[:280]}`"
                )
                await self.kobold.unload_all_models()
            else:
                await status_msg.edit_text(f"🎨 Tags generated; ComfyUI is drawing...\n`{pos_tags[:300]}`")
            png_bytes = await self._run_comfyui_txt2img(
                pos_tags, negative_tags=neg_tags, loras=loras, clip_skip=clip_skip
            )
            await bot.send_photo(
                chat_id=chat_id,
                photo=io.BytesIO(png_bytes),
                caption=f"🎨 Scene illustration\nTags: {pos_tags[:250]}",
            )
            await status_msg.delete()
        except Exception as exc:
            logger.exception("Illustration failed: %s", exc)
            await status_msg.edit_text(f"❌ Illustration failed: {exc}")

    async def cmd_illustrate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        rt = self.get_runtime(update.effective_chat.id)
        extra_hint = " ".join(context.args).strip()
        await self._perform_illustrate(
            rt=rt,
            bot=context.bot,
            chat_id=update.effective_chat.id,
            reply_to_message=update.effective_message,
            extra_hint=extra_hint,
        )

    # ------------------------------------------------------------------
    # PNG Character Card Import (Drag-and-Drop Document)
    # ------------------------------------------------------------------
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_auth(update):
            return
        doc = update.effective_message.document if update.effective_message else None
        if not doc:
            return
        fname = (doc.file_name or "").lower()
        mime = (doc.mime_type or "").lower()
        if not (fname.endswith(".png") or mime == "image/png"):
            return
        if doc.file_size and doc.file_size > 15 * 1024 * 1024:
            await update.effective_message.reply_text("❌ PNG character card exceeds the 15MB size limit.")
            return

        rt = self.get_runtime(update.effective_chat.id)
        status_msg = await update.effective_message.reply_text("📥 Parsing the PNG character card...")
        try:
            tg_file = await doc.get_file()
            raw_bytes = bytes(await tg_file.download_as_bytearray())
            char = self.storage.import_png_character(raw_bytes, fallback_name=Path(doc.file_name or "char.png").stem)
            sid = rt.register_id("chr_act", char.filename)
            lb_str = ", ".join(char.bound_lorebooks) if char.bound_lorebooks else "none"
            kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("🎭 Load Character Now (New Story)", callback_data=f"chr_new:{sid}")]
            ])
            await status_msg.edit_text(
                f"✅ Imported PNG character card: <b>{_esc(char.name)}</b>\n"
                f"• File: <code>{_esc(char.filename)}</code>\n"
                f"• Bound lorebooks: {_esc(lb_str)}",
                parse_mode=ParseMode.HTML,
                reply_markup=kb,
            )
        except Exception as exc:
            await status_msg.edit_text(f"❌ Failed to parse the PNG character card: {exc}")
