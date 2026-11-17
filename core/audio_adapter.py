"""Local CPU STT (faster-whisper) and TTS (piper-tts) adapter for PocketPortal."""

from __future__ import annotations

import asyncio
import io
import logging
import re
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("STLite.AudioAdapter")

_FEMALE_HINTS = (
    "hermione",
    "luna",
    "snow white",
    "kyoko",
    "白川",
    "ryza",
    "klaudia",
    "alice",
    "emma",
    "ella",
    "女",
)


class AudioAdapter:
    """Handles CPU-based speech-to-text and text-to-speech via ffmpeg + OGG/Opus."""

    def __init__(self, bot_config: dict[str, Any], base_dir: Path) -> None:
        self.bot_config = bot_config
        self.base_dir = Path(base_dir)
        self.audio_dir = self.base_dir / "audio_models"
        self.piper_dir = self.audio_dir / "piper"
        self.whisper_dir = self.audio_dir / "whisper"
        self.piper_dir.mkdir(parents=True, exist_ok=True)
        self.whisper_dir.mkdir(parents=True, exist_ok=True)

        self.stt_model_name: str = str(bot_config.get("stt_model", "base.en"))
        self.stt_cpu_threads: int = int(bot_config.get("stt_cpu_threads", 4))
        self.tts_default_voice: str = str(bot_config.get("tts_default_voice", "en_GB-alan-medium"))
        self.tts_female_voice: str = str(bot_config.get("tts_female_voice", "en_GB-cori-medium"))
        self.tts_narrator_voice: str = str(bot_config.get("tts_narrator_voice", "en_GB-alba-medium"))
        self.tts_dialogue_only: bool = bool(bot_config.get("tts_dialogue_only", False))

        self._whisper_model: Any = None
        self._piper_voices: dict[str, Any] = {}

    def _get_whisper(self) -> Any:
        if self._whisper_model is None:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading WhisperModel %s on CPU (int8, threads=%d)...",
                self.stt_model_name,
                self.stt_cpu_threads,
            )
            self._whisper_model = WhisperModel(
                self.stt_model_name,
                device="cpu",
                compute_type="int8",
                cpu_threads=self.stt_cpu_threads,
                download_root=str(self.whisper_dir),
            )
        return self._whisper_model

    def _pick_voice_name(self, char_name: str = "") -> str:
        low = (char_name or "").lower()
        if any(hint in low for hint in _FEMALE_HINTS):
            female_onnx = self.piper_dir / f"{self.tts_female_voice}.onnx"
            if female_onnx.exists():
                return self.tts_female_voice
        return self.tts_default_voice

    def _get_piper_voice(self, voice_name: str) -> Any:
        if voice_name not in self._piper_voices:
            from piper.voice import PiperVoice

            onnx_path = self.piper_dir / f"{voice_name}.onnx"
            json_path = self.piper_dir / f"{voice_name}.onnx.json"
            if not onnx_path.exists():
                raise FileNotFoundError(f"Piper voice model not found: {onnx_path}")
            logger.info("Loading PiperVoice %s...", voice_name)
            self._piper_voices[voice_name] = PiperVoice.load(
                str(onnx_path),
                config_path=str(json_path) if json_path.exists() else None,
            )
        return self._piper_voices[voice_name]

    @staticmethod
    def clean_text_for_speech(text: str, dialogue_only: bool = False) -> str:
        """Strip thinking tags, coach notes, and markdown punctuation for natural TTS."""
        cleaned = re.sub(r"(?is)<think>[\s\S]*?</think>", " ", text or "")
        cleaned = re.sub(r"(?im)^\s*\[💡[^\n]*\]\s*$", " ", cleaned)
        cleaned = re.sub(r"(?is)\[OOC:[^\]]*\]", " ", cleaned)
        if dialogue_only:
            quotes = re.findall(r'["“]([^"”]+)["”]', cleaned)
            if quotes:
                cleaned = " ".join(quotes)
        cleaned = re.sub(r"[`*_#>~|]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def segment_speech_elements(cls, text: str) -> list[tuple[str, str]]:
        """Segment text into narration and character dialogue tuples: (type, cleaned_text).

        type is either 'dialogue' or 'narration'.
        """
        cleaned = re.sub(r"(?is)<think>[\s\S]*?</think>", " ", text or "")
        cleaned = re.sub(r"(?im)^\s*\[💡[^\n]*\]\s*$", " ", cleaned)
        cleaned = re.sub(r"(?is)\[OOC:[^\]]*\]", " ", cleaned)

        # Match standard or curly quotes
        pattern = r'("[^"]+"|[“][^”]+[”])'
        parts = re.split(pattern, cleaned)
        segments: list[tuple[str, str]] = []

        for p in parts:
            p_strip = p.strip()
            if not p_strip:
                continue
            is_quote = (
                (p_strip.startswith('"') and p_strip.endswith('"'))
                or (p_strip.startswith('“') and p_strip.endswith('”'))
            )
            if is_quote:
                inner = p_strip[1:-1].strip()
                inner_cleaned = cls.clean_text_for_speech(inner)
                if inner_cleaned:
                    segments.append(("dialogue", inner_cleaned))
            else:
                narr_cleaned = cls.clean_text_for_speech(p_strip)
                if narr_cleaned:
                    segments.append(("narration", narr_cleaned))

        return segments

    def _transcribe_ogg_sync(self, ogg_bytes: bytes) -> str:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", "pipe:0", "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"],
            input=ogg_bytes,
            capture_output=True,
            check=True,
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(proc.stdout)
            tmp.flush()
            model = self._get_whisper()
            segments, _info = model.transcribe(tmp.name, beam_size=3, vad_filter=True)
            text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        return text.strip()

    async def transcribe_ogg(self, ogg_bytes: bytes) -> str:
        """Convert Telegram voice/audio bytes to 16kHz mono WAV and transcribe on CPU."""
        return await asyncio.to_thread(self._transcribe_ogg_sync, ogg_bytes)

    def _pick_narrator_voice_name(self) -> str:
        narrator_onnx = self.piper_dir / f"{self.tts_narrator_voice}.onnx"
        if narrator_onnx.exists():
            return self.tts_narrator_voice
        return self.tts_default_voice

    def _synthesize_ogg_sync(self, text: str, char_name: str = "") -> bytes:
        segments = self.segment_speech_elements(text)
        if not segments:
            raise ValueError("No speakable text available after cleaning.")

        char_voice_name = self._pick_voice_name(char_name)
        narr_voice_name = self._pick_narrator_voice_name()

        # If dialogue_only mode is requested, filter out narration
        if self.tts_dialogue_only:
            segments = [s for s in segments if s[0] == "dialogue"]
            if not segments:
                raise ValueError("No dialogue found for speech synthesis.")

        combined_frames = bytearray()
        target_params: Optional[wave._wave_params] = None
        pause_frames = 0

        for seg_type, seg_text in segments:
            chosen_voice = narr_voice_name if seg_type == "narration" else char_voice_name
            try:
                voice = self._get_piper_voice(chosen_voice)
            except Exception:
                voice = self._get_piper_voice(self.tts_default_voice)

            seg_buf = io.BytesIO()
            with wave.open(seg_buf, "wb") as wav_file:
                if hasattr(voice, "synthesize_wav"):
                    voice.synthesize_wav(seg_text, wav_file)
                else:
                    voice.synthesize(seg_text, wav_file)

            seg_buf.seek(0)
            with wave.open(seg_buf, "rb") as seg_wav:
                params = seg_wav.getparams()
                if target_params is None:
                    target_params = params
                    pause_frames = int(params.framerate * params.nchannels * params.sampwidth * 0.25)
                elif combined_frames and pause_frames > 0:
                    combined_frames.extend(b"\x00" * pause_frames)
                combined_frames.extend(seg_wav.readframes(seg_wav.getnframes()))

        if not combined_frames or target_params is None:
            raise ValueError("Failed to synthesize speech audio frames.")

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as final_wav:
            final_wav.setparams(target_params)
            final_wav.writeframes(combined_frames)

        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "wav",
                "-i",
                "pipe:0",
                "-c:a",
                "libopus",
                "-b:a",
                "48k",
                "-vbr",
                "on",
                "-compression_level",
                "10",
                "-application",
                "voip",
                "-f",
                "ogg",
                "pipe:1",
            ],
            input=wav_buf.getvalue(),
            capture_output=True,
            check=True,
        )
        return proc.stdout

    async def synthesize_ogg(self, text: str, char_name: str = "") -> bytes:
        """Synthesize text to Telegram-native OGG Opus voice bytes on CPU."""
        return await asyncio.to_thread(self._synthesize_ogg_sync, text, char_name)
