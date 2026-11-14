#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic ComfyUI prompt assembler and workflow builder for PocketPortal."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("PocketPortal.Illust")

HERE = Path(__file__).resolve().parent
PROMPTS_DIR = HERE / "prompts"

DEFAULT_CKPT = "model.safetensors"
DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, "
    "signature, watermark, username, blurry, 3d, realistic"
)

PERSONAS_FILE = HERE / "personas" / "personas.yaml"

# Neutral fallback personas. Real character personas belong in
# `illust/personas/personas.yaml` (git-ignored) so a public checkout never
# ships third-party characters or personal LoRA file names.
DEFAULT_PERSONA: dict[str, Any] = {
    "pos": [],
    "neg": [],
    "lora": [],
    "clip_skip": 2,
}

_load_warned = False


def _persona_config_path() -> Optional[Path]:
    """Return the optional user persona YAML path (may not exist)."""
    override = os.environ.get("ST_PERSONAS_FILE")
    if override:
        return Path(override).expanduser()
    return PERSONAS_FILE


def load_personas() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Load user personas and the name->persona map from YAML.

    Returns `(personas, name_map)`. A missing file simply yields empty dicts so
    the bot keeps working with neutral defaults.
    """
    global _load_warned
    path = _persona_config_path()
    if path is None or not path.is_file():
        return {}, {}
    try:
        import yaml  # local import: optional dependency for this feature
    except ImportError:  # pragma: no cover - requirements.txt ships pyyaml
        if not _load_warned:
            logger.warning("pyyaml not installed; skipping persona file %s", path)
            _load_warned = True
        return {}, {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 - never crash a running bot on config
        logger.warning("Failed to read personas file %s: %s", path, exc)
        return {}, {}
    personas = raw.get("personas") or {}
    name_map = {str(k).lower(): str(v) for k, v in (raw.get("name_map") or {}).items()}
    return personas, name_map


def resolve_persona_for_character(char_name: str) -> Optional[str]:
    """Map a character card name to a persona key defined in the persona file."""
    personas, name_map = load_personas()
    low = (char_name or "").lower()
    if not low:
        return None
    for key, cfg in personas.items():
        for kw in cfg.get("keywords") or []:
            if str(kw).lower() in low:
                return key
    for name, key in name_map.items():
        if name and name in low:
            return key
    return None


def load_prompt_file(path: Path) -> str:
    """Read a prompt text file, ignoring `#` / `//` comments and joining lines."""
    if not path.exists():
        return ""
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        lines.append(s)
    return ", ".join(lines)


def parse_sections(path: Path) -> dict[str, str]:
    """Parse `=== key ===` sectioned prompt library files (cloth, scene, positions, etc.)."""
    if not path.exists():
        return {}
    sections: dict[str, list[str]] = {}
    cur: Optional[str] = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s.startswith("===") and s.endswith("==="):
            cur = s.strip("= ").strip()
            sections.setdefault(cur, [])
        elif cur is not None and s and not s.startswith("#") and not s.startswith("//"):
            sections[cur].append(s)
    return {k: ", ".join(v) for k, v in sections.items() if v}


def _dedupe_tags(tag_str: str) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for raw in tag_str.split(","):
        t = re.sub(r"\s+", " ", raw.strip())
        if not t:
            continue
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return ", ".join(out)


def assemble_persona_prompt(
    persona_key: Optional[str],
    dynamic_tags: str = "",
    cloth: str = "",
    scene: str = "",
    pose: str = "",
    expr: str = "",
    extra: str = "",
) -> tuple[str, str, list[tuple[str, float]], int]:
    """Assemble (positive, negative, loras, clip_skip) in 6-layer CLIP priority order."""
    personas, _ = load_personas()
    p_cfg = personas.get(persona_key or "", {})
    pos_parts: list[str] = []
    neg_parts: list[str] = []

    for fname in p_cfg.get("pos", []):
        txt = load_prompt_file(PROMPTS_DIR / fname)
        if txt:
            pos_parts.append(txt)
    for fname in p_cfg.get("neg", []):
        txt = load_prompt_file(PROMPTS_DIR / fname)
        if txt:
            neg_parts.append(txt)

    if not pos_parts:
        pos_parts.append("masterpiece, best quality, very aesthetic, absurdres")
    if not neg_parts:
        neg_parts.append(DEFAULT_NEGATIVE)

    section_map = [
        ("cloth.txt", cloth),
        ("scene.txt", scene),
        ("positions.txt", pose),
        ("expression.txt", expr),
        ("extra.txt", extra),
    ]
    for lib_file, keys_csv in section_map:
        if not keys_csv:
            continue
        sec = parse_sections(PROMPTS_DIR / lib_file)
        for k in [x.strip() for x in keys_csv.split(",") if x.strip()]:
            pos_parts.append(sec.get(k, k))

    if dynamic_tags.strip():
        pos_parts.append(dynamic_tags.strip())

    positive = _dedupe_tags(", ".join(pos_parts))
    negative = _dedupe_tags(", ".join(neg_parts))
    loras = list(p_cfg.get("lora") or [])
    clip_skip = int(p_cfg.get("clip_skip", 2))
    return positive, negative, loras, clip_skip


def build_comfyui_workflow(
    positive: str,
    negative: str,
    ckpt: str = DEFAULT_CKPT,
    loras: Optional[list[tuple[str, float]]] = None,
    clip_skip: int = 2,
    width: int = 640,
    height: int = 896,
    steps: int = 28,
    cfg: float = 7.0,
    sampler: str = "euler_ancestral",
    scheduler: str = "normal",
    seed: Optional[int] = None,
    filename_prefix: str = "stlite_illust",
) -> dict[str, Any]:
    """Build a ComfyUI API workflow supporting LoRA stacking and CLIPSetLastLayer."""
    use_seed = seed if seed is not None else random.randint(1, 2**31 - 1)
    wf: dict[str, Any] = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt},
        }
    }
    model_ref: list[Any] = ["1", 0]
    clip_ref: list[Any] = ["1", 1]
    nid = 2

    for lora_name, strength in loras or []:
        wf[str(nid)] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_ref,
                "clip": clip_ref,
                "lora_name": lora_name,
                "strength_model": float(strength),
                "strength_clip": float(strength),
            },
        }
        model_ref, clip_ref = [str(nid), 0], [str(nid), 1]
        nid += 1

    wf[str(nid)] = {
        "class_type": "CLIPSetLastLayer",
        "inputs": {"clip": clip_ref, "stop_at_clip_layer": -abs(int(clip_skip))},
    }
    clip_ref = [str(nid), 0]
    nid += 1

    pos_id = str(nid)
    wf[pos_id] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_ref, "text": positive}}
    nid += 1

    neg_id = str(nid)
    wf[neg_id] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_ref, "text": negative}}
    nid += 1

    latent_id = str(nid)
    wf[latent_id] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": int(width), "height": int(height), "batch_size": 1},
    }
    nid += 1

    ksampler_id = str(nid)
    wf[ksampler_id] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": use_seed,
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": sampler,
            "scheduler": scheduler,
            "denoise": 1.0,
            "model": model_ref,
            "positive": [pos_id, 0],
            "negative": [neg_id, 0],
            "latent_image": [latent_id, 0],
        },
    }
    nid += 1

    vae_id = str(nid)
    wf[vae_id] = {"class_type": "VAEDecode", "inputs": {"samples": [ksampler_id, 0], "vae": ["1", 2]}}
    nid += 1

    wf[str(nid)] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": filename_prefix, "images": [vae_id, 0]},
    }
    return wf


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic ComfyUI illustration generator")
    parser.add_argument("--persona", default="", help="Persona key (e.g. default)")
    parser.add_argument("--cloth", default="")
    parser.add_argument("--scene", default="")
    parser.add_argument("--pose", default="")
    parser.add_argument("--expr", default="")
    parser.add_argument("--extra", default="")
    parser.add_argument("-p", "--prompt", default="", help="Additional dynamic tags")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("-W", "--width", type=int, default=640)
    parser.add_argument("-H", "--height", type=int, default=896)
    parser.add_argument("-s", "--steps", type=int, default=28)
    parser.add_argument("-c", "--cfg", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pos, neg, loras, clip_skip = assemble_persona_prompt(
        persona_key=args.persona or None,
        dynamic_tags=args.prompt,
        cloth=args.cloth,
        scene=args.scene,
        pose=args.pose,
        expr=args.expr,
        extra=args.extra,
    )
    wf = build_comfyui_workflow(
        positive=pos,
        negative=neg,
        ckpt=args.ckpt,
        loras=loras,
        clip_skip=clip_skip,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
    )
    if args.dry_run:
        print(json.dumps({"positive": pos, "negative": neg, "loras": loras, "workflow": wf}, ensure_ascii=False, indent=2))
        return

    req = urllib.request.Request(
        f"{args.comfy_url}/prompt",
        data=json.dumps({"prompt": wf}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        pid = json.loads(resp.read().decode("utf-8"))["prompt_id"]
    print(f"Submitted prompt_id={pid}")
    for _ in range(120):
        time.sleep(1.5)
        with urllib.request.urlopen(f"{args.comfy_url}/history/{pid}", timeout=10) as hresp:
            hist = json.loads(hresp.read().decode("utf-8"))
        if pid in hist:
            print(json.dumps(hist[pid].get("outputs", {}), ensure_ascii=False, indent=2))
            return
    raise TimeoutError("ComfyUI generation timed out")


if __name__ == "__main__":
    main()
