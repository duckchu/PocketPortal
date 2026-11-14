"""Layer 4: Storage & State Manager for PocketPortal Telegram Roleplay Bot."""

from __future__ import annotations

import base64
import json
import os
import re
import struct
import uuid
import zlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_filename(name: str, fallback: str = "unnamed") -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|\s]+", "_", name.strip())
    cleaned = cleaned.strip("._")
    return cleaned or fallback


@dataclass
class LorebookEntry:
    uid: str | int
    keys: list[str]
    content: str
    constant: bool = False
    insertion_order: int = 100
    depth: int = 4
    position: int = 0  # 0/1 = top/before-after char, 4 = @Depth injection
    selective: bool = False
    secondary_keys: list[str] = field(default_factory=list)
    disable: bool = False
    comment: str = ""
    role: Optional[int] = None
    source: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any], idx: int = 0, source: str = "") -> "LorebookEntry":
        keys = raw.get("keys")
        if keys is None:
            keys = raw.get("key", [])
        if isinstance(keys, str):
            keys = [k.strip() for k in keys.split(",") if k.strip()]

        sec_keys = raw.get("secondary_keys")
        if sec_keys is None:
            sec_keys = raw.get("keysecondary", [])
        if isinstance(sec_keys, str):
            sec_keys = [k.strip() for k in sec_keys.split(",") if k.strip()]

        order = raw.get("insertion_order")
        if order is None:
            order = raw.get("order", 100)

        disable = bool(raw.get("disable", False))
        if raw.get("enabled") is False:
            disable = True

        ext = raw.get("extensions") if isinstance(raw.get("extensions"), dict) else {}

        raw_pos = raw.get("position")
        if raw_pos is None:
            raw_pos = ext.get("position", 0)
        if isinstance(raw_pos, str):
            pos_map = {
                "before_char": 0,
                "after_char": 1,
                "before_an": 2,
                "after_an": 3,
                "at_depth": 4,
            }
            pos_val = pos_map.get(raw_pos.strip().lower())
            if pos_val is None:
                try:
                    pos_val = int(raw_pos)
                except ValueError:
                    ext_pos = ext.get("position")
                    pos_val = int(ext_pos) if isinstance(ext_pos, int) else 0
        else:
            pos_val = int(raw_pos if raw_pos is not None else 0)

        raw_depth = raw.get("depth")
        if raw_depth is None:
            raw_depth = ext.get("depth", 4)
        depth_val = int(raw_depth if raw_depth is not None else 4)
        if pos_val != 4 and depth_val <= 0:
            depth_val = 4

        return cls(
            uid=raw.get("uid", raw.get("id", idx)),
            keys=[str(k) for k in (keys or []) if str(k).strip()],
            content=str(raw.get("content", "") or ""),
            constant=bool(raw.get("constant", False)),
            insertion_order=int(order if order is not None else 100),
            depth=depth_val,
            position=pos_val,
            selective=bool(raw.get("selective", False)),
            secondary_keys=[str(k) for k in (sec_keys or []) if str(k).strip()],
            disable=disable,
            comment=str(raw.get("comment", raw.get("name", "")) or ""),
            role=raw.get("role", ext.get("role")),
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CharacterCard:
    filename: str
    name: str
    description: str = ""
    personality: str = ""
    scenario: str = ""
    first_mes: str = ""
    mes_example: str = ""
    post_history_instructions: str = ""
    system_prompt: str = ""
    bound_lorebooks: list[str] = field(default_factory=list)
    character_book_entries: list[LorebookEntry] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    depth_prompt: Optional[dict[str, Any]] = None
    alternate_greetings: list[str] = field(default_factory=list)


@dataclass
class Turn:
    turn_id: int
    role: str  # "user" | "assistant"
    swipes: list[str] = field(default_factory=list)
    active_swipe: int = 0
    timestamp: str = field(default_factory=_now_iso)
    telegram_message_id: Optional[int] = None
    telegram_message_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.telegram_message_ids and self.telegram_message_id is not None:
            self.telegram_message_ids = [int(self.telegram_message_id)]
        elif self.telegram_message_ids and self.telegram_message_id is None:
            self.telegram_message_id = int(self.telegram_message_ids[0])

    def set_message_ids(self, msg_ids: list[int]) -> None:
        cleaned = [int(m) for m in msg_ids if m]
        self.telegram_message_ids = cleaned
        self.telegram_message_id = cleaned[0] if cleaned else None

    def get_message_ids(self) -> list[int]:
        if self.telegram_message_ids:
            return list(self.telegram_message_ids)
        if self.telegram_message_id is not None:
            return [int(self.telegram_message_id)]
        return []

    @property
    def text(self) -> str:
        if not self.swipes:
            return ""
        idx = max(0, min(self.active_swipe, len(self.swipes) - 1))
        return self.swipes[idx]

    def add_swipe(self, text: str) -> int:
        self.swipes.append(text)
        self.active_swipe = len(self.swipes) - 1
        self.timestamp = _now_iso()
        return self.active_swipe

    def set_text(self, text: str) -> None:
        if not self.swipes:
            self.swipes = [text]
            self.active_swipe = 0
        else:
            idx = max(0, min(self.active_swipe, len(self.swipes) - 1))
            self.swipes[idx] = text
        self.timestamp = _now_iso()

    def prev_swipe(self) -> bool:
        if self.active_swipe > 0:
            self.active_swipe -= 1
            return True
        return False

    def next_swipe(self) -> bool:
        if self.active_swipe < len(self.swipes) - 1:
            self.active_swipe += 1
            return True
        return False

    def to_record(self) -> dict[str, Any]:
        return {
            "type": "turn",
            "turn_id": self.turn_id,
            "role": self.role,
            "swipes": self.swipes,
            "active_swipe": self.active_swipe,
            "timestamp": self.timestamp,
            "telegram_message_id": self.telegram_message_id,
            "telegram_message_ids": self.get_message_ids(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_record(cls, rec: dict[str, Any]) -> "Turn":
        raw_ids = rec.get("telegram_message_ids")
        msg_ids = [int(m) for m in raw_ids if m] if isinstance(raw_ids, list) else []
        msg_id = rec.get("telegram_message_id")
        if not msg_ids and msg_id is not None:
            msg_ids = [int(msg_id)]
        return cls(
            turn_id=int(rec.get("turn_id", 0)),
            role=str(rec.get("role", "user")),
            swipes=list(rec.get("swipes", [])),
            active_swipe=int(rec.get("active_swipe", 0)),
            timestamp=str(rec.get("timestamp", _now_iso())),
            telegram_message_id=msg_id,
            telegram_message_ids=msg_ids,
            metadata=dict(rec.get("metadata", {})),
        )


@dataclass
class SessionState:
    session_id: str
    char_filename: str
    char_name: str
    file_path: Path
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    title: str = ""
    active_model: str = ""
    bound_lorebooks: list[str] = field(default_factory=list)
    temp_lore_entries: list[dict[str, Any]] = field(default_factory=list)
    plot_note: str = ""
    plot_once: str = ""
    persona_override: str = ""
    style_override: str = ""
    user_name: str = "Player"
    is_checkpoint: bool = False
    turns: list[Turn] = field(default_factory=list)
    last_triggered_lore: list[str] = field(default_factory=list)
    last_token_usage: dict[str, int] = field(default_factory=dict)
    story_summary: str = ""
    summary_upto_turn_id: int = 0
    story_arc: dict[str, Any] = field(default_factory=dict)

    def next_turn_id(self) -> int:
        if not self.turns:
            return 1
        return max(t.turn_id for t in self.turns) + 1

    def append_turn(
        self,
        role: str,
        text: str,
        telegram_message_id: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Turn:
        turn = Turn(
            turn_id=self.next_turn_id(),
            role=role,
            swipes=[text],
            active_swipe=0,
            telegram_message_id=telegram_message_id,
            metadata=metadata or {},
        )
        self.turns.append(turn)
        self.updated_at = _now_iso()
        return turn

    def get_last_turn(self, role: Optional[str] = None) -> Optional[Turn]:
        for turn in reversed(self.turns):
            if role is None or turn.role == role:
                return turn
        return None

    def add_temp_lore(self, keys: list[str], content: str, depth: int = 4) -> LorebookEntry:
        uid = f"temp_{len(self.temp_lore_entries) + 1}"
        entry = LorebookEntry(
            uid=uid,
            keys=keys,
            content=content,
            constant=False,
            insertion_order=120,
            depth=depth,
            position=0,
            comment="Temporary Lore",
            source="session_temp",
        )
        self.temp_lore_entries.append(entry.to_dict())
        self.updated_at = _now_iso()
        return entry

    def get_temp_lore_entries(self) -> list[LorebookEntry]:
        return [
            LorebookEntry.from_dict(item, idx=i, source="session_temp")
            for i, item in enumerate(self.temp_lore_entries)
        ]

    def to_meta_record(self) -> dict[str, Any]:
        return {
            "type": "session_meta",
            "session_id": self.session_id,
            "char_filename": self.char_filename,
            "char_name": self.char_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "title": self.title,
            "active_model": self.active_model,
            "bound_lorebooks": self.bound_lorebooks,
            "temp_lore_entries": self.temp_lore_entries,
            "plot_note": self.plot_note,
            "plot_once": self.plot_once,
            "persona_override": self.persona_override,
            "style_override": self.style_override,
            "user_name": self.user_name,
            "is_checkpoint": self.is_checkpoint,
            "last_triggered_lore": self.last_triggered_lore,
            "last_token_usage": self.last_token_usage,
            "story_summary": self.story_summary,
            "summary_upto_turn_id": self.summary_upto_turn_id,
            "story_arc": self.story_arc,
        }


class StorageManager:
    """Manages configs, characters, lorebooks, JSONL sessions, and Markdown exports."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.config_dir = self.base_dir / "config"
        self.characters_dir = self.base_dir / "characters"
        self.lorebooks_dir = self.base_dir / "lorebooks"
        self.sessions_dir = self.base_dir / "sessions"
        self.exports_dir = self.base_dir / "exports"
        self.logs_dir = self.base_dir / "logs"

        for d in (
            self.config_dir,
            self.characters_dir,
            self.lorebooks_dir,
            self.sessions_dir,
            self.exports_dir,
            self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        self._lorebook_cache: dict[str, tuple[int, list[LorebookEntry]]] = {}

    # ------------------------------------------------------------------
    # Config & Presets
    # ------------------------------------------------------------------
    def _load_yaml(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def load_bot_config(self) -> dict[str, Any]:
        cfg = self._load_yaml(self.config_dir / "bot_config.yaml")
        env_token = os.environ.get("ST_BOT_TOKEN")
        if env_token:
            cfg["bot_token"] = env_token.strip()
        return cfg

    def load_models_config(self) -> dict[str, Any]:
        return self._load_yaml(self.config_dir / "models.yaml")

    def load_system_presets(self) -> dict[str, Any]:
        return self._load_yaml(self.config_dir / "system_presets.yaml")

    # ------------------------------------------------------------------
    # Characters
    # ------------------------------------------------------------------
    def list_characters(self) -> list[CharacterCard]:
        cards: list[CharacterCard] = []
        if not self.characters_dir.exists():
            return cards
        for p in sorted(self.characters_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in {".json", ".yaml", ".yml"}:
                try:
                    cards.append(self.load_character(p.name))
                except Exception:
                    continue
        return cards

    def _resolve_lorebook_filename(self, world_name: str) -> str:
        world_name = world_name.strip()
        if not world_name:
            return ""
        if (self.lorebooks_dir / world_name).exists():
            return world_name
        if not world_name.lower().endswith(".json"):
            candidate = f"{world_name}.json"
            if (self.lorebooks_dir / candidate).exists():
                return candidate
            return candidate
        return world_name

    def load_character(self, name_or_filename: str) -> CharacterCard:
        target = self.characters_dir / name_or_filename
        if not target.exists():
            for p in sorted(self.characters_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in {".json", ".yaml", ".yml"}:
                    if p.stem == name_or_filename or p.name == name_or_filename:
                        target = p
                        break
            if not target.exists():
                for card in self.list_characters():
                    if card.name == name_or_filename:
                        target = self.characters_dir / card.filename
                        break

        if not target.exists():
            raise FileNotFoundError(f"Character not found: {name_or_filename}")

        if target.suffix.lower() in {".yaml", ".yml"}:
            raw = self._load_yaml(target)
        else:
            with open(target, "r", encoding="utf-8") as f:
                raw = json.load(f)

        inner = raw.get("data") if isinstance(raw.get("data"), dict) else {}

        def _pick(key: str, default: Any = "") -> Any:
            val = inner.get(key)
            if val not in (None, "", []):
                return val
            val = raw.get(key)
            if val not in (None, "", []):
                return val
            return default

        extensions = _pick("extensions", {}) or {}
        if not isinstance(extensions, dict):
            extensions = {}

        bound: list[str] = []
        raw_bound = raw.get("bound_lorebooks") or inner.get("bound_lorebooks") or []
        if isinstance(raw_bound, str):
            raw_bound = [raw_bound]
        for item in raw_bound:
            resolved = self._resolve_lorebook_filename(str(item))
            if resolved and resolved not in bound:
                bound.append(resolved)

        if not bound:
            ext_world = str(extensions.get("world", "") or "").strip()
            if ext_world:
                resolved = self._resolve_lorebook_filename(ext_world)
                if resolved:
                    bound.append(resolved)

        char_book_raw = _pick("character_book", None)
        char_book_entries: list[LorebookEntry] = []
        if isinstance(char_book_raw, dict):
            raw_entries = char_book_raw.get("entries", [])
            if isinstance(raw_entries, dict):
                iter_entries = list(raw_entries.values())
            elif isinstance(raw_entries, list):
                iter_entries = raw_entries
            else:
                iter_entries = []
            for idx, item in enumerate(iter_entries):
                if isinstance(item, dict):
                    char_book_entries.append(
                        LorebookEntry.from_dict(item, idx=idx, source=f"charbook:{target.name}")
                    )

        depth_prompt = extensions.get("depth_prompt")
        if not isinstance(depth_prompt, dict):
            depth_prompt = None

        alt_raw = _pick("alternate_greetings", []) or []
        alternate_greetings = [str(x).strip() for x in alt_raw if str(x).strip()] if isinstance(alt_raw, list) else []

        return CharacterCard(
            filename=target.name,
            name=str(_pick("name", target.stem)),
            description=str(_pick("description", "")),
            personality=str(_pick("personality", "")),
            scenario=str(_pick("scenario", "")),
            first_mes=str(_pick("first_mes", "")),
            mes_example=str(_pick("mes_example", "")),
            post_history_instructions=str(_pick("post_history_instructions", "")),
            system_prompt=str(_pick("system_prompt", "")),
            bound_lorebooks=bound,
            character_book_entries=char_book_entries,
            tags=[str(t) for t in (_pick("tags", []) or [])],
            depth_prompt=depth_prompt,
            alternate_greetings=alternate_greetings,
        )

    # ------------------------------------------------------------------
    # Lorebooks
    # ------------------------------------------------------------------
    def list_lorebooks(self) -> list[str]:
        if not self.lorebooks_dir.exists():
            return []
        return sorted(
            p.name
            for p in self.lorebooks_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".json"
        )

    def load_lorebook(self, filename: str) -> list[LorebookEntry]:
        resolved = self._resolve_lorebook_filename(filename)
        path = self.lorebooks_dir / resolved
        if not path.exists():
            self._lorebook_cache.pop(resolved, None)
            return []

        mtime_ns = path.stat().st_mtime_ns
        cached = self._lorebook_cache.get(resolved)
        if cached is not None and cached[0] == mtime_ns:
            return cached[1]

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        entries_raw = raw.get("entries", [])
        if isinstance(entries_raw, dict):
            items = list(entries_raw.values())
        elif isinstance(entries_raw, list):
            items = entries_raw
        else:
            items = []

        entries = [
            LorebookEntry.from_dict(item, idx=idx, source=resolved)
            for idx, item in enumerate(items)
            if isinstance(item, dict)
        ]
        self._lorebook_cache[resolved] = (mtime_ns, entries)
        return entries

    @staticmethod
    def extract_chara_from_png(png_bytes: bytes) -> dict[str, Any]:
        """Parse SillyTavern V2 character JSON from PNG tEXt/zTXt/iTXt 'chara' chunk."""
        if not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("File is not a valid PNG image.")
        pos = 8
        total = len(png_bytes)
        chara_b64: Optional[str] = None
        while pos + 8 <= total:
            length = struct.unpack(">I", png_bytes[pos : pos + 4])[0]
            ctype = png_bytes[pos + 4 : pos + 8]
            data_start = pos + 8
            data_end = data_start + length
            if data_end > total:
                break
            chunk_data = png_bytes[data_start:data_end]
            pos = data_end + 4  # skip CRC

            if ctype == b"tEXt":
                if b"\x00" in chunk_data:
                    kw, val = chunk_data.split(b"\x00", 1)
                    if kw.decode("latin-1", errors="ignore").strip().lower() == "chara":
                        chara_b64 = val.decode("utf-8", errors="ignore").strip()
                        break
            elif ctype == b"zTXt":
                if b"\x00" in chunk_data:
                    kw, rest = chunk_data.split(b"\x00", 1)
                    if kw.decode("latin-1", errors="ignore").strip().lower() == "chara" and len(rest) >= 1:
                        comp_data = rest[1:]
                        chara_b64 = zlib.decompress(comp_data).decode("utf-8", errors="ignore").strip()
                        break
            elif ctype == b"iTXt":
                parts = chunk_data.split(b"\x00", 5)
                if len(parts) >= 6 and parts[0].decode("latin-1", errors="ignore").strip().lower() == "chara":
                    comp_flag = parts[1][:1] == b"\x01"
                    raw_text = zlib.decompress(parts[5]) if comp_flag else parts[5]
                    chara_b64 = raw_text.decode("utf-8", errors="ignore").strip()
                    break
            elif ctype == b"IEND":
                break

        if not chara_b64:
            raise ValueError("No SillyTavern 'chara' character data block found in this PNG.")
        raw_json = base64.b64decode(chara_b64).decode("utf-8")
        data = json.loads(raw_json)
        if not isinstance(data, dict):
            raise ValueError("The chara JSON inside the PNG is malformed.")
        return data

    def import_png_character(self, png_bytes: bytes, fallback_name: str = "imported_char") -> CharacterCard:
        raw = self.extract_chara_from_png(png_bytes)
        data = raw.get("data") if isinstance(raw.get("data"), dict) else raw
        char_name = str(data.get("name") or raw.get("name") or fallback_name).strip()
        filename = f"{_safe_filename(char_name, 'imported_char')}.json"
        out_path = self.characters_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
        return self.load_character(filename)

    # ------------------------------------------------------------------
    # Sessions (JSONL)
    # ------------------------------------------------------------------
    def _sessions_dir_for_chat(self, chat_id: Optional[int | str] = None) -> Path:
        if chat_id is None:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            return self.sessions_dir
        d = self.sessions_dir / str(chat_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def create_session(
        self,
        character: CharacterCard,
        active_model: str,
        user_name: str = "Player",
        include_first_mes: bool = True,
        chat_id: Optional[int] = None,
    ) -> SessionState:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:6]
        safe_char = _safe_filename(character.name, "char")
        path = self._sessions_dir_for_chat(chat_id) / f"{safe_char}_{ts}_{short_id}.jsonl"

        session = SessionState(
            session_id=f"{ts}_{short_id}",
            char_filename=character.filename,
            char_name=character.name,
            file_path=path,
            title=f"{character.name} New Story",
            active_model=active_model,
            bound_lorebooks=list(character.bound_lorebooks),
            user_name=user_name,
        )
        if include_first_mes and character.first_mes.strip():
            turn0 = session.append_turn(role="assistant", text=character.first_mes.strip())
            for alt in character.alternate_greetings:
                if alt.strip() and alt.strip() != character.first_mes.strip():
                    turn0.swipes.append(alt.strip())
            turn0.active_swipe = 0
        self.save_session(session)
        return session

    def save_session(self, session: SessionState) -> Path:
        session.updated_at = _now_iso()
        tmp_path = session.file_path.with_suffix(".jsonl.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(session.to_meta_record(), ensure_ascii=False) + "\n")
            for turn in session.turns:
                f.write(json.dumps(turn.to_record(), ensure_ascii=False) + "\n")
        os.replace(tmp_path, session.file_path)
        return session.file_path

    def load_session(self, path: str | Path) -> SessionState:
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            p = self.sessions_dir / p
        if not p.exists():
            raise FileNotFoundError(f"Session file not found: {p}")

        meta: dict[str, Any] = {}
        turns: list[Turn] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rtype = rec.get("type")
                if rtype == "session_meta":
                    meta = rec
                elif rtype == "turn":
                    turns.append(Turn.from_record(rec))

        if not meta:
            raise ValueError(f"Invalid session JSONL (missing session_meta): {p}")

        return SessionState(
            session_id=str(meta.get("session_id", p.stem)),
            char_filename=str(meta.get("char_filename", "")),
            char_name=str(meta.get("char_name", "")),
            file_path=p,
            created_at=str(meta.get("created_at", _now_iso())),
            updated_at=str(meta.get("updated_at", _now_iso())),
            title=str(meta.get("title", "")),
            active_model=str(meta.get("active_model", "")),
            bound_lorebooks=list(meta.get("bound_lorebooks", [])),
            temp_lore_entries=list(meta.get("temp_lore_entries", [])),
            plot_note=str(meta.get("plot_note", "")),
            plot_once=str(meta.get("plot_once", "")),
            persona_override=str(meta.get("persona_override", "")),
            style_override=str(meta.get("style_override", "")),
            user_name=str(meta.get("user_name", "Player")),
            is_checkpoint=bool(meta.get("is_checkpoint", False)),
            turns=turns,
            last_triggered_lore=list(meta.get("last_triggered_lore", [])),
            last_token_usage=dict(meta.get("last_token_usage", {})),
            story_summary=str(meta.get("story_summary", "")),
            summary_upto_turn_id=int(meta.get("summary_upto_turn_id", 0)),
            story_arc=dict(meta.get("story_arc") or {}),
        )

    def list_sessions(
        self, char_name: Optional[str] = None, limit: int = 20, chat_id: Optional[int] = None
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        target_dir = self._sessions_dir_for_chat(chat_id)
        if not target_dir.exists():
            return results
        for p in target_dir.glob("*.jsonl"):
            try:
                sess = self.load_session(p)
                if char_name and sess.char_name != char_name:
                    continue
                last_turn = sess.get_last_turn()
                preview = last_turn.text.replace("\n", " ")[:48] if last_turn else "(No messages yet)"
                results.append({
                    "path": p,
                    "filename": p.name,
                    "session_id": sess.session_id,
                    "char_name": sess.char_name,
                    "char_filename": sess.char_filename,
                    "updated_at": sess.updated_at,
                    "title": sess.title,
                    "is_checkpoint": sess.is_checkpoint,
                    "turn_count": len(sess.turns),
                    "preview": preview,
                })
            except Exception:
                continue
        results.sort(key=lambda item: item["updated_at"], reverse=True)
        return results[:limit]

    def get_latest_session_for_char(
        self, char_name: str, chat_id: Optional[int] = None
    ) -> Optional[SessionState]:
        items = self.list_sessions(char_name=char_name, limit=1, chat_id=chat_id)
        if not items:
            return None
        return self.load_session(items[0]["path"])

    def create_checkpoint(self, session: SessionState, custom_title: str = "") -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_char = _safe_filename(session.char_name, "char")
        safe_title = _safe_filename(custom_title or "checkpoint", "checkpoint")
        cp_dir = session.file_path.parent if session.file_path else self.sessions_dir
        cp_dir.mkdir(parents=True, exist_ok=True)
        cp_path = cp_dir / f"{safe_char}_{ts}_{safe_title}.jsonl"

        cp = SessionState(
            session_id=f"{ts}_{uuid.uuid4().hex[:6]}",
            char_filename=session.char_filename,
            char_name=session.char_name,
            file_path=cp_path,
            created_at=_now_iso(),
            updated_at=_now_iso(),
            title=custom_title or f"Checkpoint ({ts})",
            active_model=session.active_model,
            bound_lorebooks=list(session.bound_lorebooks),
            temp_lore_entries=list(session.temp_lore_entries),
            plot_note=session.plot_note,
            plot_once=session.plot_once,
            persona_override=session.persona_override,
            style_override=session.style_override,
            user_name=session.user_name,
            is_checkpoint=True,
            turns=[Turn.from_record(t.to_record()) for t in session.turns],
            last_triggered_lore=list(session.last_triggered_lore),
            last_token_usage=dict(session.last_token_usage),
            story_summary=session.story_summary,
            summary_upto_turn_id=session.summary_upto_turn_id,
            story_arc=json.loads(json.dumps(session.story_arc)) if session.story_arc else {},
        )
        return self.save_session(cp)

    # ------------------------------------------------------------------
    # Story Markdown Export
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_story_text(text: str) -> str:
        cleaned = re.sub(r"(?is)\[(?:System Note|OOC|Author's Note|Scene Advance)[^\]]*\]", "", text)
        cleaned = re.sub(r"(?is)\*\(\s*🎬\s*Scene Advance[:：][^\)]*\)\*", "", cleaned)
        cleaned = re.sub(r"(?is)\(\s*OOC\s*[:：][^)]*\)", "", cleaned)
        cleaned = re.sub(r"\s*//\s*OOC\b.*$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        return cleaned.strip()

    def export_story_markdown(
        self,
        session: SessionState,
        character: Optional[CharacterCard] = None,
        custom_title: str = "",
    ) -> Path:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_char = _safe_filename(session.char_name, "char")
        safe_title = _safe_filename(custom_title or "story", "story")
        out_path = self.exports_dir / f"{safe_char}_{date_str}_{safe_title}.md"

        title_display = custom_title.strip() or f"{session.char_name} Adventure Log"
        now_display = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines: list[str] = [
            f"# {title_display}",
            "",
            f"- **Character**: {session.char_name}",
            f"- **Protagonist**: {session.user_name}",
            f"- **Exported At**: {now_display}",
            f"- **Lorebooks**: {', '.join(session.bound_lorebooks) if session.bound_lorebooks else 'None'}",
        ]
        if session.plot_note:
            lines.append(f"- **Plot Note**: {session.plot_note}")
        if session.persona_override:
            lines.append(f"- **Persona Override**: {session.persona_override}")
        if character and character.scenario:
            lines.extend([
                "",
                "## Scenario",
                "",
                character.scenario.replace("{{char}}", session.char_name).replace("{{user}}", session.user_name),
            ])
        if session.story_arc and session.story_arc.get("title"):
            arc = session.story_arc
            lines.extend([
                "",
                f"## 🗺️ Story Arc: {arc.get('title', '')}",
                "",
            ])
            if arc.get("premise"):
                lines.append(f"- **Premise**: {arc['premise']}")
            if arc.get("divergence_note"):
                lines.append(f"- **Divergence Note**: {arc['divergence_note']}")
            chapters = arc.get("chapters") or []
            cur_idx = int(arc.get("current_chapter", 0))
            for idx, ch in enumerate(chapters):
                mark = "📍" if idx == cur_idx else ("✅" if idx < cur_idx else "⏳")
                ch_name = ch.get("name", f"Chapter {idx + 1}")
                ch_goal = ch.get("goal", "")
                lines.append(f"- {mark} **{ch_name}**：{ch_goal}")

        lines.extend(["", "---", "", "## Story", ""])

        for turn in session.turns:
            clean_text = self._clean_story_text(turn.text)
            if not clean_text:
                continue
            if turn.role == "user":
                lines.append(f"**{session.user_name}**：")
                lines.append("")
                lines.append(clean_text)
                lines.append("")
            else:
                lines.append(f"**{session.char_name}**：")
                lines.append("")
                lines.append(clean_text)
                lines.append("")

        out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return out_path
