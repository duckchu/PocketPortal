"""Lightweight JSON-based localisation helper.

English is the default language baked into the source. Locale files under
`i18n/` map the English string back to a translated string, so callers can do:

    from core.i18n import t
    await message.reply_text(t("⏳ Generating reply..."))

Missing keys fall back to the English source string. Set the language with the
`ST_LOCALE` environment variable (default: `en`).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("PocketPortal.i18n")

I18N_DIR = Path(__file__).resolve().parent.parent / "i18n"
DEFAULT_LOCALE = "en"

_cache: dict[str, dict[str, str]] = {}
_lang: Optional[str] = None


def available_locales() -> list[str]:
    """Return every locale code that has a JSON file under `i18n/`."""
    if not I18N_DIR.is_dir():
        return [DEFAULT_LOCALE]
    return sorted(p.stem for p in I18N_DIR.glob("*.json"))


def current_locale() -> str:
    global _lang
    if _lang is None:
        _lang = (os.environ.get("ST_LOCALE") or DEFAULT_LOCALE).strip() or DEFAULT_LOCALE
    return _lang


def set_locale(locale: str) -> None:
    """Switch the active locale at runtime."""
    global _lang
    _lang = (locale or DEFAULT_LOCALE).strip() or DEFAULT_LOCALE
    gettext(_lang)  # warm the cache


def gettext(locale: str) -> dict[str, str]:
    """Load and cache the translation table for `locale`."""
    if locale in _cache:
        return _cache[locale]
    path = I18N_DIR / f"{locale}.json"
    table: dict[str, str] = {}
    if path.is_file():
        try:
            table = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - a broken locale must not crash the bot
            logger.warning("Failed to load locale %s: %s", locale, exc)
    elif locale != DEFAULT_LOCALE:
        logger.warning("Locale %s not found in %s", locale, I18N_DIR)
    _cache[locale] = table
    return table


def t(text: str, locale: Optional[str] = None) -> str:
    """Translate an English UI string into the active locale."""
    lang = locale or current_locale()
    if lang == DEFAULT_LOCALE:
        return text
    return gettext(lang).get(text, text)
