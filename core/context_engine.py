"""Layer 2: ST-Lite Context Engine (8-Stage Prompt Stack, Lorebook Scanner, OOC & Regex)."""

from __future__ import annotations

import hashlib
import html
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from core.kobold_adapter import KoboldAdapter
from core.rich_renderer import convert_markdown_to_telegram_html
from core.storage import CharacterCard, LorebookEntry, SessionState, StorageManager


@dataclass
class LoreScanResult:
    top_entries: list[LorebookEntry] = field(default_factory=list)
    depth_entries: list[LorebookEntry] = field(default_factory=list)
    triggered_keys: list[str] = field(default_factory=list)


@dataclass
class AssembledPrompt:
    prompt_text: str
    prompt_tokens: int
    max_context: int
    triggered_lore_keys: list[str] = field(default_factory=list)
    included_turns_count: int = 0
    stop_sequences: list[str] = field(default_factory=list)
    instruct_template: str = "chatml"
    sections_debug: dict[str, str] = field(default_factory=dict)


class ContextEngine:
    """SillyTavern-compatible Context Assembly & Post-Processing Engine."""

    def __init__(
        self,
        storage: StorageManager,
        kobold: KoboldAdapter,
        bot_config: dict[str, Any],
        presets_config: dict[str, Any],
    ) -> None:
        self.storage = storage
        self.kobold = kobold
        self.bot_config = bot_config
        self.presets_config = presets_config
        self._opencc: Any = None
        if bool(bot_config.get("force_traditional_chinese", True)):
            try:
                import opencc
                self._opencc = opencc.OpenCC("s2twp")
            except Exception:
                self._opencc = None

    # ------------------------------------------------------------------
    # Macros & OOC Parsing
    # ------------------------------------------------------------------
    @staticmethod
    def replace_macros(text: str, char_name: str, user_name: str) -> str:
        if not text:
            return ""
        out = re.sub(r"\{\{\s*char\s*\}\}|<BOT>", char_name, text, flags=re.IGNORECASE)
        out = re.sub(r"\{\{\s*user\s*\}\}|<USER>", user_name, out, flags=re.IGNORECASE)

        def _repl_random(m: re.Match[str]) -> str:
            raw = m.group(1)
            items = raw.split("::") if "::" in raw else raw.split(",")
            choices = [i.strip() for i in items if i.strip()]
            return random.choice(choices) if choices else ""

        def _repl_roll(m: re.Match[str]) -> str:
            spec = m.group(1).strip().lower()
            dice_m = re.fullmatch(r"(\d*)d(\d+)", spec)
            if dice_m:
                n_dice = min(20, max(1, int(dice_m.group(1) or "1")))
                sides = max(2, int(dice_m.group(2)))
                return str(sum(random.randint(1, sides) for _ in range(n_dice)))
            if spec.isdigit():
                sides = max(2, int(spec))
                return str(random.randint(1, sides))
            return m.group(0)

        out = re.sub(r"\{\{\s*random\s*(?:::|:)(.*?)\}\}", _repl_random, out, flags=re.IGNORECASE | re.DOTALL)
        out = re.sub(r"\{\{\s*roll\s*(?:::|:)([^}]+)\}\}", _repl_roll, out, flags=re.IGNORECASE)
        return out

    @staticmethod
    def parse_user_input(raw_text: str) -> tuple[str, Optional[str]]:
        """
        Splits inline OOC directives from normal dialogue.
        Supports:
          - Trailing `(OOC: ...)` / `（OOC：...）`
          - Trailing `// ...` (not preceded by `:` such as `https://`)
        Returns `(clean_dialogue, inline_ooc)`.
        """
        text = raw_text.strip()
        if not text:
            return "", None

        ooc_parts: list[str] = []

        # Extract any (OOC: ...) or （OOC：...） segments
        paren_pattern = re.compile(r"[\(（]\s*OOC\s*[:：]\s*(.*?)[\)）]", re.IGNORECASE | re.DOTALL)
        matches = list(paren_pattern.finditer(text))
        if matches:
            for m in matches:
                note = m.group(1).strip()
                if note:
                    ooc_parts.append(note)
            text = paren_pattern.sub("", text).strip()

        # Check trailing // OOC note (avoid URL schemes like http:// or https://)
        slash_match = re.search(r"(?:^|(?<!:)\s+)//\s*(.+)$", text, flags=re.DOTALL)
        if slash_match:
            note = slash_match.group(1).strip()
            if note:
                ooc_parts.append(note)
            text = text[: slash_match.start()].strip()

        inline_ooc = " ; ".join(ooc_parts) if ooc_parts else None
        return text, inline_ooc

    # ------------------------------------------------------------------
    # Lorebook Scanning
    # ------------------------------------------------------------------
    @staticmethod
    def _match_key(key: str, haystack: str) -> bool:
        key = key.strip()
        if not key:
            return False
        regex_m = re.fullmatch(r"/(.+)/([imsgu]*)", key, flags=re.DOTALL)
        if regex_m:
            pattern, flags_str = regex_m.group(1), regex_m.group(2)
            flags = 0
            if "i" in flags_str:
                flags |= re.IGNORECASE
            if "m" in flags_str:
                flags |= re.MULTILINE
            if "s" in flags_str:
                flags |= re.DOTALL
            try:
                return bool(re.search(pattern, haystack, flags=flags))
            except re.error:
                return False
        return key.lower() in haystack.lower()

    def scan_lorebooks(
        self,
        session: SessionState,
        character: CharacterCard,
        recent_text: str,
        max_recursion_depth: int = 2,
    ) -> LoreScanResult:
        all_entries: list[LorebookEntry] = []
        # Bound world lorebooks & session temporary entries take precedence over embedded character_book
        for lb_name in session.bound_lorebooks:
            all_entries.extend(self.storage.load_lorebook(lb_name))
        all_entries.extend(session.get_temp_lore_entries())
        all_entries.extend(character.character_book_entries)

        top_entries: list[LorebookEntry] = []
        depth_entries: list[LorebookEntry] = []
        triggered_keys: list[str] = []
        seen_contents: set[str] = set()
        pending_entries: list[LorebookEntry] = []

        # Pass 0: constant entries + initial keyword scan against recent_text
        newly_triggered_texts: list[str] = []
        for entry in all_entries:
            if entry.disable or not entry.content.strip():
                continue

            matched_label: Optional[str] = None
            if entry.constant:
                matched_label = entry.comment or (entry.keys[0] if entry.keys else "constant entry")
            else:
                matched_primary = [k for k in entry.keys if self._match_key(k, recent_text)]
                if not matched_primary:
                    pending_entries.append(entry)
                    continue
                if entry.selective and entry.secondary_keys:
                    if not any(self._match_key(sk, recent_text) for sk in entry.secondary_keys):
                        pending_entries.append(entry)
                        continue
                matched_label = matched_primary[0]

            norm_content = entry.content.strip()
            if norm_content in seen_contents:
                continue
            seen_contents.add(norm_content)
            newly_triggered_texts.append(norm_content)

            if matched_label and matched_label not in triggered_keys:
                triggered_keys.append(matched_label)

            if entry.constant or entry.position == 1:
                top_entries.append(entry)
            elif entry.position == 4 or entry.position == 0:
                depth_entries.append(entry)
            else:
                top_entries.append(entry)

        # Recursive passes (up to max_recursion_depth): scan triggered entry contents
        for _ in range(max_recursion_depth):
            if not newly_triggered_texts or not pending_entries:
                break
            recursive_corpus = "\n".join(newly_triggered_texts)
            newly_triggered_texts = []
            next_pending: list[LorebookEntry] = []
            for entry in pending_entries:
                matched_primary = [k for k in entry.keys if self._match_key(k, recursive_corpus)]
                if not matched_primary:
                    next_pending.append(entry)
                    continue
                if entry.selective and entry.secondary_keys:
                    combined_corpus = f"{recent_text}\n{recursive_corpus}"
                    if not any(self._match_key(sk, combined_corpus) for sk in entry.secondary_keys):
                        next_pending.append(entry)
                        continue
                norm_content = entry.content.strip()
                if norm_content in seen_contents:
                    continue
                seen_contents.add(norm_content)
                newly_triggered_texts.append(norm_content)

                matched_label = matched_primary[0]
                if matched_label and matched_label not in triggered_keys:
                    triggered_keys.append(matched_label)

                if entry.constant or entry.position == 1:
                    top_entries.append(entry)
                else:
                    depth_entries.append(entry)
            pending_entries = next_pending

        top_entries.sort(key=lambda e: (e.insertion_order, str(e.uid)))
        depth_entries.sort(key=lambda e: (e.depth, e.insertion_order, str(e.uid)), reverse=True)
        return LoreScanResult(
            top_entries=top_entries,
            depth_entries=depth_entries,
            triggered_keys=triggered_keys,
        )

    # ------------------------------------------------------------------
    # Instruct Formatting Helpers
    # ------------------------------------------------------------------
    def _get_template(self, template_name: str) -> dict[str, Any]:
        templates = self.presets_config.get("instruct_templates", {})
        if template_name in templates:
            return templates[template_name]
        return templates.get("chatml", {
            "system_prefix": "<|im_start|>system\n",
            "system_suffix": "<|im_end|>\n",
            "user_prefix": "<|im_start|>user\n",
            "user_suffix": "<|im_end|>\n",
            "assistant_prefix": "<|im_start|>assistant\n",
            "assistant_suffix": "<|im_end|>\n",
            "assistant_prefill": "<|im_start|>assistant\n",
            "stop_sequences": ["<|im_end|>"],
        })

    @staticmethod
    def _wrap_block(role: str, content: str, tpl: dict[str, Any]) -> str:
        content = content.strip()
        if not content:
            return ""
        if role == "system":
            prefix = tpl.get("system_prefix", "")
            suffix = tpl.get("system_suffix", "\n")
            return f"{prefix}{content}{suffix}"
        if role == "user":
            prefix = tpl.get("user_prefix", "")
            suffix = tpl.get("user_suffix", "\n")
            return f"{prefix}{content}{suffix}"
        prefix = tpl.get("assistant_prefix", "")
        suffix = tpl.get("assistant_suffix", "\n")
        return f"{prefix}{content}{suffix}"

    def _format_examples(
        self,
        mes_example: str,
        char_name: str,
        user_name: str,
        tpl: dict[str, Any],
    ) -> str:
        if not mes_example.strip():
            return ""
        cleaned = self.replace_macros(mes_example.strip(), char_name, user_name)
        blocks = [b.strip() for b in re.split(r"<START>", cleaned, flags=re.IGNORECASE) if b.strip()]
        rendered_parts: list[str] = []
        for idx, block in enumerate(blocks):
            rendered_parts.append(f"[Example Dialogue {idx + 1}]\n{block}")
        return self._wrap_block("system", "\n\n".join(rendered_parts), tpl)

    # ------------------------------------------------------------------
    # 8-Stage Prompt Stack Builder
    # ------------------------------------------------------------------
    async def build_prompt(
        self,
        session: SessionState,
        character: CharacterCard,
        model_key: Optional[str] = None,
        pending_user_text: Optional[str] = None,
        inline_ooc: Optional[str] = None,
        director_instruction: Optional[str] = None,
        ooc_question: Optional[str] = None,
        rewrite_guidance: Optional[str] = None,
        exclude_last_assistant: bool = False,
        continue_mode: bool = False,
    ) -> AssembledPrompt:
        m_key = model_key or session.active_model or self.kobold.active_model_key
        model_info = self.kobold.get_model_info(m_key)
        max_context = int(model_info.get("context_size", 8192))
        max_gen = int((model_info.get("sampling") or {}).get("max_length", 744))
        tpl_name = str(model_info.get("instruct_template", "chatml"))
        tpl = self._get_template(tpl_name)

        preset_key = str(self.bot_config.get("active_preset", "neutral_rp"))
        presets = self.presets_config.get("presets", {})
        preset = presets.get(preset_key, next(iter(presets.values()), {}))

        char_name = character.name
        user_name = session.user_name or str(self.bot_config.get("default_user_name", "Player"))

        base_turns = list(session.turns)
        if exclude_last_assistant and base_turns and base_turns[-1].role == "assistant":
            base_turns.pop()

        scan_chunks = [t.text for t in base_turns[-6:]]
        if pending_user_text:
            scan_chunks.append(pending_user_text)
        if director_instruction:
            scan_chunks.append(director_instruction)
        arc = session.story_arc if isinstance(session.story_arc, dict) else {}
        arc_enabled = bool(arc and arc.get("enabled", True) and (arc.get("title") or arc.get("premise") or arc.get("chapters")))
        arc_chapters = list(arc.get("chapters") or []) if arc_enabled else []
        arc_cur_idx = max(0, min(int(arc.get("current_chapter", 0)), len(arc_chapters) - 1)) if arc_chapters else 0
        if arc_enabled:
            if arc.get("title"):
                scan_chunks.append(str(arc["title"]))
            if arc.get("premise"):
                scan_chunks.append(str(arc["premise"]))
            if arc_chapters:
                cur_ch = arc_chapters[arc_cur_idx]
                scan_chunks.append(f"{cur_ch.get('name', '')} {cur_ch.get('goal', '')} {cur_ch.get('hooks', '')}")
        recent_text = "\n".join(scan_chunks)
        lore_scan = self.scan_lorebooks(session, character, recent_text)

        # ---- Stage 1: Top (System Prompt + User Persona + Character Card) ----
        stage1_parts: list[str] = []
        sys_prompt = str(preset.get("system_prompt", "")).strip()
        if sys_prompt:
            stage1_parts.append(self.replace_macros(sys_prompt, char_name, user_name))

        user_persona = str(self.bot_config.get("default_user_persona", "")).strip()
        if user_persona:
            stage1_parts.append(f"[User Persona]\n{self.replace_macros(user_persona, char_name, user_name)}")

        if character.system_prompt.strip():
            stage1_parts.append(self.replace_macros(character.system_prompt.strip(), char_name, user_name))
        if character.description.strip():
            stage1_parts.append(f"[{char_name} Description]\n{self.replace_macros(character.description.strip(), char_name, user_name)}")
        if character.personality.strip():
            stage1_parts.append(f"[{char_name} Personality]\n{self.replace_macros(character.personality.strip(), char_name, user_name)}")
        if character.scenario.strip():
            stage1_parts.append(f"[Scenario]\n{self.replace_macros(character.scenario.strip(), char_name, user_name)}")
        if character.post_history_instructions.strip():
            stage1_parts.append(
                f"[Roleplay Rules]\n{self.replace_macros(character.post_history_instructions.strip(), char_name, user_name)}"
            )
        if character.depth_prompt and str(character.depth_prompt.get("prompt", "")).strip():
            stage1_parts.append(
                self.replace_macros(str(character.depth_prompt["prompt"]).strip(), char_name, user_name)
            )

        stage1_raw = "\n\n".join(stage1_parts)

        # ---- Stage 2: Pre-knowledge (Constant & High-Priority Lorebooks) ----
        stage2_parts: list[str] = []
        for entry in lore_scan.top_entries:
            label = entry.comment or (", ".join(entry.keys[:2]) if entry.keys else "Lore")
            content = self.replace_macros(entry.content.strip(), char_name, user_name)
            stage2_parts.append(f"[World Info: {label}]\n{content}")
        if session.story_summary.strip():
            stage2_parts.append(
                f"[Story Summary]\n{self.replace_macros(session.story_summary.strip(), char_name, user_name)}"
            )
        if arc_enabled:
            ch_lines: list[str] = []
            for idx, ch in enumerate(arc_chapters):
                marker = "📍[Current Chapter]" if idx == arc_cur_idx else ("✅" if idx < arc_cur_idx else "⏳")
                ch_lines.append(f"  {marker} {ch.get('name', f'Chapter {idx + 1}')}: {ch.get('goal', '')}")
            arc_text = f"[Story Campaign Outline: {arc.get('title', 'Untitled Campaign')}]\nPremise: {arc.get('premise', '')}"
            if ch_lines:
                arc_text += "\nChapters:\n" + "\n".join(ch_lines)
            if arc.get("divergence_note"):
                arc_text += f"\nCurrent Branch Evolution: {arc['divergence_note']}"
            stage2_parts.append(self.replace_macros(arc_text, char_name, user_name))
        stage2_raw = "\n\n".join(stage2_parts)

        top_system_combined = stage1_raw
        if stage2_raw:
            top_system_combined = f"{stage1_raw}\n\n{stage2_raw}" if stage1_raw else stage2_raw
        bos = str(tpl.get("bos_token", ""))
        stage12_block = bos + self._wrap_block("system", top_system_combined, tpl)

        system_top_only = bool(tpl.get("system_top_only", False))

        # ---- Stage 3: Tone Anchor (Example Chat) ----
        stage3_block = "" if system_top_only else self._format_examples(character.mes_example, char_name, user_name, tpl)
        if system_top_only and character.mes_example.strip():
            ex_cleaned = self.replace_macros(character.mes_example.strip(), char_name, user_name)
            ex_blocks = [b.strip() for b in re.split(r"<START>", ex_cleaned, flags=re.IGNORECASE) if b.strip()]
            ex_text = "\n\n".join(f"[Example Dialogue {i + 1}]\n{b}" for i, b in enumerate(ex_blocks))
            if ex_text:
                top_system_combined = f"{top_system_combined}\n\n{ex_text}" if top_system_combined else ex_text
                stage12_block = bos + self._wrap_block("system", top_system_combined, tpl)

        # ---- Stage 5 (@Depth 4): Dynamic Persona & Style Override ----
        stage5_parts: list[str] = []
        if session.persona_override.strip():
            stage5_parts.append(
                f"[Dynamic Persona Override]\n{self.replace_macros(session.persona_override.strip(), char_name, user_name)}"
            )
        if session.style_override.strip():
            stage5_parts.append(
                f"[Output Style & Tone Override]\n{self.replace_macros(session.style_override.strip(), char_name, user_name)}"
            )
        for entry in lore_scan.depth_entries:
            if entry.depth >= 3:
                label = entry.comment or "Lore"
                stage5_parts.append(
                    f"[Context Lore: {label}]\n{self.replace_macros(entry.content.strip(), char_name, user_name)}"
                )
        stage5_raw = "\n\n".join(stage5_parts)
        stage5_block = "" if system_top_only else (self._wrap_block("system", stage5_raw, tpl) if stage5_raw else "")

        # ---- Stage 6 (@Depth 2): Plot / Author's Note + Depth<=2 Lore + Global Post-History ----
        stage6_parts: list[str] = []
        post_hist = str(preset.get("post_history_instructions", "")).strip()
        if post_hist:
            stage6_parts.append(self.replace_macros(post_hist, char_name, user_name))
        if session.plot_note.strip():
            stage6_parts.append(
                f"[Author's Note / Plot Direction]\n{self.replace_macros(session.plot_note.strip(), char_name, user_name)}"
            )
        if arc_enabled and arc_chapters:
            cur_ch = arc_chapters[arc_cur_idx]
            ch_name = cur_ch.get("name", f"Chapter {arc_cur_idx + 1}")
            ch_goal = cur_ch.get("goal", "")
            ch_hooks = cur_ch.get("hooks", "")
            arc_guide = (
                f"[Story Arc Guidance - {ch_name}]\n"
                f"• Current Chapter Context & Hooks: {ch_goal}"
                + (f" (Hooks/Clues: {ch_hooks}）\n" if ch_hooks else "\n")
                + f"• Flexible Branching Rule (Important): Treat the outline above as background events and foreshadowing only. Prioritize {user_name}'s actual actions and choices—if {user_name} follows the hooks, advance naturally; if {user_name} diverges, follow their choices into organic side-branches and butterfly effects without forcing them back onto the original rails."
            )
            stage6_parts.append(self.replace_macros(arc_guide, char_name, user_name))
        for entry in lore_scan.depth_entries:
            if entry.depth < 3:
                label = entry.comment or "Detail Guide"
                stage6_parts.append(
                    f"[Scene Guidance: {label}]\n{self.replace_macros(entry.content.strip(), char_name, user_name)}"
                )
        stage6_raw = "\n\n".join(stage6_parts)
        stage6_block = "" if system_top_only else (self._wrap_block("system", stage6_raw, tpl) if stage6_raw else "")

        # ---- Stage 7: Terminal Instruction (Latest Input + OOC / Director / Rewrite) ----
        terminal_role: Optional[str] = None
        terminal_content = ""

        history_for_stage4 = [
            t for t in base_turns if t.turn_id > session.summary_upto_turn_id
        ]
        continue_prefill = ""
        if continue_mode and history_for_stage4 and history_for_stage4[-1].role == "assistant":
            last_a = history_for_stage4.pop()
            continue_prefill = self.replace_macros(last_a.text.strip(), char_name, user_name)

        if pending_user_text is not None:
            terminal_role = "user"
            terminal_content = self.replace_macros(pending_user_text.strip(), char_name, user_name)
        elif director_instruction:
            terminal_role = "user"
            terminal_content = (
                f"[System Note / Director Instruction: "
                f"{self.replace_macros(director_instruction.strip(), char_name, user_name)}]"
            )
        elif ooc_question:
            terminal_role = "user"
            terminal_content = (
                "[OOC Mode - Out of Character Discussion: Temporarily pause the roleplay and answer "
                "the user's OOC question concisely in Traditional Chinese based on the story context.]\n"
                f"{self.replace_macros(ooc_question.strip(), char_name, user_name)}"
            )
        elif history_for_stage4 and history_for_stage4[-1].role == "user":
            last_u = history_for_stage4.pop()
            terminal_role = "user"
            raw_u = last_u.text.strip()
            terminal_content = "" if raw_u == "[Scene Advance]" else self.replace_macros(raw_u, char_name, user_name)

        ooc_notes: list[str] = []
        if continue_prefill:
            ooc_notes.append("[System Note: Continue directly from the previous sentence without repeating prior text.]")
        if not ooc_question and session.plot_once.strip():
            ooc_notes.append(
                f"[Immediate Event for This Turn: {self.replace_macros(session.plot_once.strip(), char_name, user_name)}]"
            )
        if inline_ooc and inline_ooc.strip():
            ooc_notes.append(f"[System Note / OOC Instruction: {self.replace_macros(inline_ooc.strip(), char_name, user_name)}]")
        if rewrite_guidance and rewrite_guidance.strip():
            ooc_notes.append(f"[System Note / Rewrite Guidance: {self.replace_macros(rewrite_guidance.strip(), char_name, user_name)}]")
        if not ooc_question:
            overused = self._extract_overused_phrases(history_for_stage4, char_name)
            if overused:
                ooc_notes.append(f"[Avoid repeating these recently used phrases or tropes: {'、'.join(overused)}]")
            term_inst = str(preset.get("terminal_instruction", "")).strip()
            if term_inst:
                ooc_notes.append(self.replace_macros(term_inst, char_name, user_name))

        if system_top_only:
            merged_user_parts = [p for p in (stage5_raw, stage6_raw, terminal_content, "\n".join(ooc_notes)) if p]
            merged_user = "\n\n".join(merged_user_parts)
            stage7_block = self._wrap_block("user", merged_user, tpl) if merged_user else ""
        else:
            stage7_user_block = (
                self._wrap_block("user", terminal_content, tpl) if (terminal_role and terminal_content) else ""
            )
            stage7_sys_block = self._wrap_block("system", "\n".join(ooc_notes), tpl) if ooc_notes else ""
            stage7_block = stage7_user_block + stage7_sys_block

        # ---- Stage 8: Assistant Prefill ----
        jailbreak_prefill = str(preset.get("jailbreak_prefill", "")).strip()
        if ooc_question:
            stage8_block = str(tpl.get("assistant_prefix", "<|im_start|>assistant\n"))
        elif continue_prefill:
            base_start = str(tpl.get("assistant_prefill") or tpl.get("assistant_prefix", "<|im_start|>assistant\n"))
            stage8_block = base_start + continue_prefill
        else:
            stage8_block = str(tpl.get("assistant_prefill", "<|im_start|>assistant\n"))
            if jailbreak_prefill:
                stage8_block += self.replace_macros(jailbreak_prefill, char_name, user_name)

        # ---- Budget Allocation for Stage 3 (Examples) & Stage 4 (History) ----
        safety_margin = 64
        budget_total = max(512, max_context - max_gen - safety_margin)

        fixed_text = stage12_block + stage5_block + stage6_block + stage7_block + stage8_block
        fixed_tokens = await self.kobold.count_tokens(fixed_text)
        remaining_tokens = max(0, budget_total - fixed_tokens)

        include_stage3 = False
        if stage3_block:
            s3_tokens = await self.kobold.count_tokens(stage3_block)
            if s3_tokens <= remaining_tokens and (len(history_for_stage4) <= 4 or remaining_tokens - s3_tokens >= 600):
                include_stage3 = True
                remaining_tokens -= s3_tokens
            else:
                stage3_block = ""

        selected_turns: list[tuple[str, str]] = []
        for turn in reversed(history_for_stage4):
            t_text = self.replace_macros(turn.text.strip(), char_name, user_name)
            wrapped = self._wrap_block(turn.role, t_text, tpl)
            if not wrapped:
                continue
            text_hash = hashlib.md5(wrapped.encode("utf-8")).hexdigest()[:12]
            t_cache = turn.metadata.get("token_cache")
            cached_entry = t_cache.get(tpl_name) if isinstance(t_cache, dict) else None
            if isinstance(cached_entry, dict) and cached_entry.get("h") == text_hash:
                t_tokens = int(cached_entry["t"])
            else:
                t_tokens = self.kobold.estimate_tokens_fallback(wrapped)
                turn.metadata.setdefault("token_cache", {})[tpl_name] = {
                    "h": text_hash,
                    "t": t_tokens,
                }
            if t_tokens <= remaining_tokens:
                selected_turns.append((turn.role, wrapped))
                remaining_tokens -= t_tokens
            else:
                break
        selected_turns.reverse()

        # Interleave Stage 5 (@Depth 4) and Stage 6 (@Depth 2) relative to total chat tail
        tail_count = len(selected_turns) + (1 if stage7_block else 0)
        idx_d4 = max(0, tail_count - 4)
        idx_d2 = max(0, tail_count - 2)

        body_blocks: list[str] = []
        d4_inserted = False
        d2_inserted = False
        for i, (_, wrapped_turn) in enumerate(selected_turns):
            if stage5_block and not d4_inserted and i >= idx_d4:
                body_blocks.append(stage5_block)
                d4_inserted = True
            if stage6_block and not d2_inserted and i >= idx_d2:
                body_blocks.append(stage6_block)
                d2_inserted = True
            body_blocks.append(wrapped_turn)

        if stage5_block and not d4_inserted:
            body_blocks.append(stage5_block)
        if stage6_block and not d2_inserted:
            body_blocks.append(stage6_block)

        full_prompt = (
            stage12_block
            + (stage3_block if include_stage3 else "")
            + "".join(body_blocks)
            + stage7_block
            + stage8_block
        )

        total_prompt_tokens = await self.kobold.count_tokens(full_prompt)

        stop_seqs = list(tpl.get("stop_sequences") or ["<|im_end|>"])
        system_leak_stops = [
            "\nSystem Prompt",
            "System Prompt",
            "```mdSystem Prompt",
            "\n[System Prompt]",
            "\n[System Prompt",
            "\n[System Note",
            "\n[Highest Priority Rule",
            "\n[Roleplay Rules]",
            "(END_OF_TURN)",
            "(END_OF_TURN)",
            "(WAITING_FOR_USER_INPUT",
            "<|thought|",
            "| [User Persona",
            "| [Constraint Check",
            "| [Next Step",
            "| [Output Generated",
            "|end_of_turn>",
            "<end_of_turn>",
            "token_count:",
            "__(:",
            "*(🎬 Scene Advance",
        ]
        for s in system_leak_stops:
            if s not in stop_seqs:
                stop_seqs.append(s)
        if user_name:
            extra_stops = [
                f"\n{user_name}:",
                f"\n{user_name}：",
                f"\n【{user_name}】",
                f"\n{user_name} looked at",
                f"\n{user_name} thought",
                f"\n{user_name} felt",
                f"\n{user_name} could not help",
                f"\n{user_name} tried to",
                f"\n{user_name} attempted",
                f"\n{user_name} said",
                f"\n{user_name} spoke",
            ]
            for s in extra_stops:
                if s not in stop_seqs:
                    stop_seqs.append(s)

        return AssembledPrompt(
            prompt_text=full_prompt,
            prompt_tokens=total_prompt_tokens,
            max_context=max_context,
            triggered_lore_keys=lore_scan.triggered_keys,
            included_turns_count=len(selected_turns) + (1 if stage7_block else 0),
            stop_sequences=stop_seqs,
            instruct_template=tpl_name,
            sections_debug={
                "stage1_2": stage12_block,
                "stage3": stage3_block,
                "stage4_5_6": "".join(body_blocks),
                "stage7": stage7_block,
                "stage8": stage8_block,
            },
        )

    @staticmethod
    def _extract_overused_phrases(turns: list[Any], char_name: str = "") -> list[str]:
        assistant_turns = [t for t in turns if getattr(t, "role", "") == "assistant"][-3:]
        if not assistant_turns:
            return []
        counts: Counter[str] = Counter()
        for t in assistant_turns:
            txt = getattr(t, "text", "") or ""
            clauses = re.split(r"[，。！？!?；;\n「」『』\*]+", txt)
            for c in clauses:
                c = c.strip()
                if 6 <= len(c) <= 28 and c != char_name:
                    counts[c] += 1
        return [phrase for phrase, cnt in counts.most_common(6) if cnt >= 2]

    @staticmethod
    def _deduplicate_sentences(text: str) -> str:
        if not text:
            return ""
        seen_actions: set[str] = set()

        def _dedup_action(match: re.Match[str]) -> str:
            inner = match.group(1).strip()
            if len(inner) >= 10:
                if inner in seen_actions:
                    return ""
                seen_actions.add(inner)
            return match.group(0)

        text = re.sub(r"\*([^*\n]+)\*", _dedup_action, text)

        seen_sentences: set[str] = set()
        out_paras: list[str] = []
        for para in text.split("\n"):
            if not para.strip():
                out_paras.append(para)
                continue
            parts = re.findall(r"[^。！？!?\n]+[。！？!?]*[」』\"'）)\*]*", para)
            if not parts:
                parts = [para]
            kept: list[str] = []
            for seg in parts:
                norm = seg.strip()
                if not norm:
                    continue
                if len(norm) >= 12:
                    if norm in seen_sentences:
                        continue
                    seen_sentences.add(norm)
                kept.append(seg)
            joined = "".join(kept).strip()
            if joined:
                out_paras.append(joined)
        return "\n".join(out_paras)

    # ------------------------------------------------------------------
    # Regex Post-Processing & Telegram Formatting
    # ------------------------------------------------------------------
    def post_process_output(
        self,
        raw_text: str,
        char_name: str = "",
        user_name: str = "",
        is_preview: bool = False,
    ) -> str:
        text = raw_text or ""
        text = re.sub(r"(?is)(?:<think>[\s\S]*?</think>|<\|channel>thought[\s\S]*?<channel\|>)\s*", "", text)
        if is_preview:
            low = text.lower()
            if "<think>" in low and "</think>" not in low:
                text = re.sub(r"(?is)<think>.*$", "\n💭 *(Thinking...)*", text)
            if "<|channel>thought" in low and "<channel|>" not in low:
                text = re.sub(r"(?is)<\|channel>thought.*$", "\n💭 *(Thinking...)*", text)

        rules = self.presets_config.get("regex_post_process", [])
        for rule in rules:
            pattern = rule.get("pattern")
            repl = rule.get("replacement", "")
            if not pattern:
                continue
            try:
                text = re.sub(pattern, repl, text)
            except re.error:
                continue

        if not is_preview:
            text = re.sub(r"(?is)(?:<think>|<\|channel>thought).*$", "", text)

        text = re.sub(r"(?is)(?:<end\.thought\|>|<turn\|>|<\|turn>|<(?:endrefold|/?p)\s*/?>)", "", text)

        if char_name:
            text = re.sub(rf"^\s*{re.escape(char_name)}\s*[:：]\s*", "", text)
        if user_name:
            cut = re.search(
                rf"\n\s*(?:{re.escape(user_name)}\s*[:：]|{re.escape(user_name)}(?:looked at|thought|felt|could not help|tried to|attempted|said|spoke)|【{re.escape(user_name)}】)",
                text,
            )
            if cut:
                text = text[: cut.start()]

        if self._opencc is not None and text:
            try:
                text = self._opencc.convert(text)
            except Exception:
                pass

        text = re.sub(
            r"(?is)```[a-zA-Z0-9_-]*\s*(?:System Prompt|\[System Prompt\]|\[System Note|\[Highest Priority|\[Turn Rule).*?(?:```|$)",
            "",
            text,
        )
        text = re.sub(
            r"(?is)(?:\n|\r|^)\s*(?:System Prompt|\[System Prompt\]|\[System Note|\[Highest Priority|\[Turn Rule).*$",
            "",
            text,
        )
        text = re.sub(
            r"(?is)(?:\s*\|\s*)?(?:\(END_OF_TURN\)|\(WAITING_FOR_USER_INPUT).*$",
            "",
            text,
        )
        text = re.sub(r"(?:__+\s+__+\s*\)\s*_*\s*(?:\([^)\n]{0,40}\)\s*)*)+", "", text)
        text = re.sub(
            r"(?is)(?:__\(:|<\|thought\|text\|>|\|\s*\[(?:User Persona|Constraint Check|Next Step|Output Generated)|token_count\s*:|\|end_of_turn>).*$",
            "",
            text,
        )
        text = re.sub(r"(?is)\*\(\s*🎬\s*Scene (?:Advance|pushed)[:：][^\)]*\)\*", "", text)

        text = self._deduplicate_sentences(text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def format_for_telegram_html(text: str) -> str:
        """Convert Markdown / rich text into Telegram Bot API ParseMode.HTML."""
        return convert_markdown_to_telegram_html(text, native_tables=False)
