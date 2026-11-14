"""Layer 3: KoboldCpp Adapter & Model Manager Wrapper."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

import httpx
import psutil

logger = logging.getLogger("STLite.Kobold")


class KoboldAdapter:
    """Client and process manager for KoboldCpp Native API."""

    def __init__(
        self,
        bot_config: dict[str, Any],
        models_config: dict[str, Any],
        logs_dir: str | Path,
    ) -> None:
        self.base_url = str(bot_config.get("koboldcpp_url", "http://127.0.0.1:5001")).rstrip("/")
        self.ollama_url = str(bot_config.get("ollama_url", "http://127.0.0.1:11434")).rstrip("/")
        self.ollama_cloud_url = str(bot_config.get("ollama_cloud_url", "https://ollama.com")).rstrip("/")
        self.ollama_api_key = str(bot_config.get("ollama_api_key") or os.environ.get("OLLAMA_API_KEY") or "").strip()
        self.kobold_bin = str(bot_config.get("koboldcpp_bin", "/path/to/bin/koboldcpp"))
        self.port = int(bot_config.get("koboldcpp_port", 5001))
        self.mlx_port = int(bot_config.get("mlx_port", 8080))
        self.mlx_url = str(bot_config.get("mlx_url", f"http://127.0.0.1:{self.mlx_port}")).rstrip("/")
        self.mlx_python = str(bot_config.get("mlx_python", sys.executable))
        self.startup_timeout = int(bot_config.get("koboldcpp_startup_timeout", 120))
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.models: dict[str, dict[str, Any]] = dict(models_config.get("models", {}))
        default_key = str(bot_config.get("default_model") or models_config.get("default_model") or "")
        if default_key not in self.models and self.models:
            default_key = next(iter(self.models))
        self.active_model_key: str = default_key

        self._process: Optional[subprocess.Popen] = None
        self._log_file = None
        self._mlx_process: Optional[subprocess.Popen] = None
        self._mlx_log_file = None
        self._switch_lock = asyncio.Lock()
        self._gen_lock = asyncio.Lock()
        self.last_activity_ts: float = time.monotonic()
        self.last_prompt_eval_count: Optional[int] = None
        self.last_eval_count: Optional[int] = None

    def touch_activity(self) -> None:
        self.last_activity_ts = time.monotonic()

    # ------------------------------------------------------------------
    # Model Metadata
    # ------------------------------------------------------------------
    def list_models(self) -> dict[str, dict[str, Any]]:
        return self.models

    def get_model_info(self, model_key: Optional[str] = None) -> dict[str, Any]:
        key = model_key or self.active_model_key
        if key not in self.models:
            raise KeyError(f"Unknown model key: {key}")
        return self.models[key]

    def get_active_instruct_template(self) -> str:
        if self.active_model_key in self.models:
            return str(self.models[self.active_model_key].get("instruct_template", "chatml"))
        return "chatml"

    def get_active_context_size(self) -> int:
        if self.active_model_key in self.models:
            return int(self.models[self.active_model_key].get("context_size", 8192))
        return 8192

    def get_backend(self, model_key: Optional[str] = None) -> str:
        key = model_key or self.active_model_key
        if key in self.models:
            return str(self.models[key].get("backend", "koboldcpp")).lower()
        return "koboldcpp"

    def _is_ollama_backend(self, model_key: Optional[str] = None) -> bool:
        return self.get_backend(model_key) in ("ollama", "ollama_cloud")

    def _ollama_endpoint_and_headers(self, model_key: Optional[str] = None) -> tuple[str, dict[str, str]]:
        if self.get_backend(model_key) == "ollama_cloud" and self.ollama_api_key:
            return self.ollama_cloud_url, {"Authorization": f"Bearer {self.ollama_api_key}"}
        return self.ollama_url, {}

    # ------------------------------------------------------------------
    # Process / Model Manager Wrapper
    # ------------------------------------------------------------------
    def detect_running_kobold_pids(self) -> list[int]:
        pids: list[int] = []
        if self._process and self._process.poll() is None:
            pids.append(self._process.pid)
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                name = (proc.info.get("name") or "").lower()
                cmdline = " ".join(proc.info.get("cmdline") or []).lower()
                if "koboldcpp" in name or "koboldcpp" in cmdline:
                    if proc.info["pid"] not in pids:
                        pids.append(proc.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return pids

    def detect_running_mlx_pids(self) -> list[int]:
        pids: list[int] = []
        if self._mlx_process and self._mlx_process.poll() is None:
            pids.append(self._mlx_process.pid)
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = " ".join(proc.info.get("cmdline") or []).lower()
                if "mlx_lm" in cmdline and "server" in cmdline:
                    if proc.info["pid"] not in pids:
                        pids.append(proc.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return pids

    async def stop_process(self) -> None:
        pids = self.detect_running_kobold_pids() + [
            p for p in self.detect_running_mlx_pids() if p not in self.detect_running_kobold_pids()
        ]
        if not pids:
            return

        for pid in pids:
            try:
                os_proc = psutil.Process(pid)
                os_proc.send_signal(signal.SIGTERM)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        deadline = asyncio.get_running_loop().time() + 10.0
        while asyncio.get_running_loop().time() < deadline:
            alive = [p for p in pids if psutil.pid_exists(p)]
            if not alive:
                break
            await asyncio.sleep(0.4)

        for pid in pids:
            if psutil.pid_exists(pid):
                try:
                    psutil.Process(pid).kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
        self._process = None
        if self._mlx_log_file:
            try:
                self._mlx_log_file.close()
            except Exception:
                pass
            self._mlx_log_file = None
        self._mlx_process = None

    def build_cli_args(self, model_key: str) -> list[str]:
        cfg = self.get_model_info(model_key)
        kcpps = cfg.get("kcpps_config")
        if kcpps and Path(str(kcpps)).exists():
            cmd = [
                self.kobold_bin,
                "--config",
                str(kcpps),
                "--port",
                str(self.port),
                "--skiplauncher",
            ]
        else:
            model_path = str(cfg.get("path", ""))
            if not Path(model_path).exists():
                raise FileNotFoundError(f"GGUF model file not found: {model_path}")
            cmd = [
                self.kobold_bin,
                "--model",
                model_path,
                "--port",
                str(self.port),
                "--contextsize",
                str(cfg.get("context_size", 65536)),
                "--gpulayers",
                str(cfg.get("gpu_layers", 999)),
                "--quantkv",
                str(cfg.get("quantkv", 1)),
                "--skiplauncher",
            ]
            if cfg.get("use_vulkan", False):
                cmd.append("--usevulkan")
            if not cfg.get("flash_attention", True):
                cmd.append("--noflashattention")

        extra_args = cfg.get("extra_args") or []
        if isinstance(extra_args, list):
            cmd.extend(str(a) for a in extra_args)
        return cmd

    def build_mlx_cli_args(self, model_key: str) -> list[str]:
        cfg = self.get_model_info(model_key)
        mlx_model = str(cfg.get("mlx_model") or cfg.get("path") or "").strip()
        if not mlx_model:
            raise ValueError(f"Missing mlx_model or path for {model_key}")
        cmd = [
            self.mlx_python,
            "-m",
            "mlx_lm",
            "server",
            "--model",
            mlx_model,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.mlx_port),
        ]
        extra_args = cfg.get("extra_args") or []
        if isinstance(extra_args, list):
            cmd.extend(str(a) for a in extra_args)
        return cmd

    async def get_loaded_model_from_api(self) -> Optional[str]:
        backend = self.get_backend()
        if backend == "mlx":
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(f"{self.mlx_url}/v1/models")
                if resp.status_code == 200:
                    data = resp.json().get("data") or []
                    if data and isinstance(data[0], dict) and data[0].get("id"):
                        return f"mlx/{data[0]['id']}"
                    cfg = self.get_model_info()
                    return f"mlx/{cfg.get('mlx_model') or self.active_model_key}"
            except Exception:
                pass
            return None
        if backend == "ollama_cloud":
            cfg = self.get_model_info()
            ollama_model = str(cfg.get("ollama_model", ""))
            if self.ollama_api_key:
                return f"ollama-cloud/{ollama_model} (☁️ Cloud Ready)"
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(f"{self.ollama_url}/api/version")
                    if resp.status_code == 200:
                        return f"ollama-cloud/{ollama_model} (☁️ Cloud Ready)"
            except Exception:
                pass
            return None
        if backend == "ollama":
            cfg = self.get_model_info()
            ollama_model = str(cfg.get("ollama_model", ""))
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(f"{self.ollama_url}/api/ps")
                    if resp.status_code == 200:
                        running = [m.get("name") for m in resp.json().get("models", [])]
                        if ollama_model in running:
                            return f"ollama/{ollama_model}"
                    tags_resp = await client.get(f"{self.ollama_url}/api/tags")
                    if tags_resp.status_code == 200:
                        available = [m.get("name") for m in tags_resp.json().get("models", [])]
                        if ollama_model in available:
                            return f"ollama/{ollama_model} (Ready)"
            except Exception:
                pass
            return None

        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/api/v1/model")
            if resp.status_code == 200:
                data = resp.json()
                result = str(data.get("result", "") or "").strip()
                if result and result.lower() != "readonly":
                    return result
        except Exception:
            pass
        return None

    async def switch_model(
        self,
        model_key: str,
        progress_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> str:
        if model_key not in self.models:
            raise KeyError(f"Model key not found: {model_key}")

        async with self._switch_lock:
            backend = self.get_backend(model_key)
            if backend == "mlx":
                cmd = self.build_mlx_cli_args(model_key)
                logger.info("Stopping existing KoboldCpp/MLX and Ollama models before loading MLX %s", model_key)
                await self.stop_process()
                await self.unload_ollama_models()

                log_path = self.logs_dir / "mlx.log"
                self._mlx_log_file = open(log_path, "a", encoding="utf-8")
                self._mlx_log_file.write(f"\n=== Launching MLX {model_key}: {' '.join(cmd)} ===\n")
                self._mlx_log_file.flush()

                self._mlx_process = subprocess.Popen(
                    cmd,
                    stdout=self._mlx_log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

                deadline = asyncio.get_running_loop().time() + self.startup_timeout
                last_notify = 0.0
                while asyncio.get_running_loop().time() < deadline:
                    if self._mlx_process.poll() is not None:
                        tail = ""
                        try:
                            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-15:])
                        except Exception:
                            pass
                        raise RuntimeError(f"MLX server exited prematurely (code={self._mlx_process.returncode}):\n{tail}")

                    try:
                        async with httpx.AsyncClient(timeout=2.0) as client:
                            resp = await client.get(f"{self.mlx_url}/v1/models")
                        if resp.status_code == 200:
                            self.active_model_key = model_key
                            self.touch_activity()
                            data = resp.json().get("data") or []
                            loaded_id = data[0].get("id") if (data and isinstance(data[0], dict)) else model_key
                            logger.info("MLX model loaded: %s (%s)", model_key, loaded_id)
                            return f"mlx/{loaded_id}"
                    except Exception:
                        pass

                    now = asyncio.get_running_loop().time()
                    if progress_callback and now - last_notify > 15.0:
                        last_notify = now
                        await progress_callback(f"⏳ Loading or downloading MLX model ({model_key})...")

                    await asyncio.sleep(1.5)

                raise TimeoutError(f"Timed out waiting for MLX server to load {model_key} within {self.startup_timeout}s")

            if backend == "ollama_cloud":
                cfg = self.get_model_info(model_key)
                ollama_model = str(cfg.get("ollama_model", "")).strip()
                if not ollama_model:
                    raise ValueError(f"Missing ollama_model for {model_key}")
                logger.info("Switching to Ollama Cloud model %s; releasing local KoboldCpp/Ollama VRAM", ollama_model)
                await self.stop_process()
                await self.unload_ollama_models()
                if not self.ollama_api_key:
                    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=10.0)) as client:
                        tags_resp = await client.get(f"{self.ollama_url}/api/tags")
                        tags_resp.raise_for_status()
                        available = [m.get("name") for m in tags_resp.json().get("models", [])]
                        if ollama_model not in available:
                            pull_resp = await client.post(
                                f"{self.ollama_url}/api/pull",
                                json={"model": ollama_model, "stream": False},
                            )
                            pull_resp.raise_for_status()
                self.active_model_key = model_key
                self.touch_activity()
                logger.info("Ollama Cloud model ready: %s (%s)", model_key, ollama_model)
                return f"ollama-cloud/{ollama_model}"

            if backend == "ollama":
                cfg = self.get_model_info(model_key)
                ollama_model = str(cfg.get("ollama_model", "")).strip()
                if not ollama_model:
                    raise ValueError(f"Missing ollama_model for {model_key}")
                logger.info("Switching to Ollama model %s; stopping KoboldCpp and other Ollama models", ollama_model)
                await self.stop_process()
                loaded_ollama = await self.get_loaded_ollama_models()
                ctx_size = int(cfg.get("context_size", 65536))
                async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=self.startup_timeout, write=10.0, pool=10.0)) as client:
                    for other_model in loaded_ollama:
                        if other_model != ollama_model and "embedding" not in other_model.lower():
                            try:
                                await client.post(
                                    f"{self.ollama_url}/api/generate",
                                    json={"model": other_model, "prompt": "", "keep_alive": 0},
                                )
                            except Exception as exc:
                                logger.warning("Failed to unload previous Ollama model %s: %s", other_model, exc)
                    tags_resp = await client.get(f"{self.ollama_url}/api/tags")
                    tags_resp.raise_for_status()
                    available = [m.get("name") for m in tags_resp.json().get("models", [])]
                    if ollama_model not in available:
                        raise RuntimeError(f"Ollama model `{ollama_model}` not found. Check `ollama list`.")
                    # Warm-load model with target num_ctx to avoid context reallocation on first turn
                    warm_resp = await client.post(
                        f"{self.ollama_url}/api/generate",
                        json={
                            "model": ollama_model,
                            "prompt": "",
                            "keep_alive": "15m",
                            "options": {"num_ctx": ctx_size},
                        },
                    )
                    warm_resp.raise_for_status()
                self.active_model_key = model_key
                self.touch_activity()
                logger.info("Ollama model ready: %s (%s)", model_key, ollama_model)
                return f"ollama/{ollama_model}"

            cmd = self.build_cli_args(model_key)
            logger.info("Stopping existing KoboldCpp and Ollama models before loading %s", model_key)
            await self.stop_process()
            await self.unload_ollama_models()

            log_path = self.logs_dir / "koboldcpp.log"
            self._log_file = open(log_path, "a", encoding="utf-8")
            cmd_line = " ".join(cmd)
            self._log_file.write(f"\n=== Launching {model_key}: {cmd_line} ===\n")
            self._log_file.flush()

            self._process = subprocess.Popen(
                cmd,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

            deadline = asyncio.get_running_loop().time() + self.startup_timeout
            last_notify = 0.0
            while asyncio.get_running_loop().time() < deadline:
                if self._process.poll() is not None:
                    tail = ""
                    try:
                        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-15:])
                    except Exception:
                        pass
                    raise RuntimeError(f"KoboldCpp exited prematurely (code={self._process.returncode}):\n{tail}")

                loaded = await self.get_loaded_model_from_api()
                if loaded:
                    self.active_model_key = model_key
                    self.touch_activity()
                    logger.info("KoboldCpp model loaded: %s (%s)", model_key, loaded)
                    return loaded

                now = asyncio.get_running_loop().time()
                if progress_callback and now - last_notify > 15.0:
                    last_notify = now
                    await progress_callback(f"⏳ Loading model ({model_key})...")

                await asyncio.sleep(1.5)

            raise TimeoutError(f"Timed out waiting for KoboldCpp to load {model_key} within {self.startup_timeout}s")

    async def ensure_running(self) -> str:
        backend = self.get_backend()
        if backend in ("ollama", "ollama_cloud"):
            if self._process is not None:
                await self.stop_process()
            cfg = self.get_model_info()
            prefix = "ollama-cloud" if backend == "ollama_cloud" else "ollama"
            return f"{prefix}/{cfg.get('ollama_model', '')}"
        loaded = await self.get_loaded_model_from_api()
        if loaded:
            return loaded
        return await self.switch_model(self.active_model_key)

    async def get_loaded_ollama_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.ollama_url}/api/ps")
            if resp.status_code == 200:
                return [str(m.get("name")) for m in resp.json().get("models", []) if m.get("name")]
        except Exception:
            pass
        return []

    async def unload_ollama_models(self) -> list[str]:
        running = await self.get_loaded_ollama_models()
        unloaded: list[str] = []
        if not running:
            return unloaded
        async with httpx.AsyncClient(timeout=10.0) as client:
            for name in running:
                try:
                    resp = await client.post(
                        f"{self.ollama_url}/api/generate",
                        json={"model": name, "prompt": "", "keep_alive": 0},
                    )
                    if resp.status_code == 200:
                        unloaded.append(name)
                except Exception as exc:
                    logger.warning("Failed to unload Ollama model %s: %s", name, exc)
        return unloaded

    async def unload_all_models(self) -> dict[str, list[Any]]:
        kobold_pids = self.detect_running_kobold_pids()
        if kobold_pids:
            await self.stop_process()
        ollama_unloaded = await self.unload_ollama_models()
        return {
            "kobold_stopped": kobold_pids,
            "ollama_unloaded": ollama_unloaded,
        }

    async def has_any_loaded_model(self) -> bool:
        if self.detect_running_kobold_pids() or self.detect_running_mlx_pids():
            return True
        if await self.get_loaded_ollama_models():
            return True
        return False

    # ------------------------------------------------------------------
    # Token Counting & Generation API
    # ------------------------------------------------------------------
    @staticmethod
    def estimate_tokens_fallback(text: str) -> int:
        if not text:
            return 0
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))
        non_cjk = re.sub(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+", " ", text)
        words = len(non_cjk.split())
        punct = len(re.findall(r"[^\w\s]", non_cjk))
        est = int(round(cjk_chars * 1.35 + words * 1.3 + punct * 0.4))
        return max(1, est)

    async def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._is_ollama_backend() or self.get_backend() == "mlx":
            return self.estimate_tokens_fallback(text)
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/extra/tokencount",
                    json={"prompt": text, "special": False},
                )
            if resp.status_code == 200:
                data = resp.json()
                if "value" in data:
                    return int(data["value"])
        except Exception:
            pass
        return self.estimate_tokens_fallback(text)

    async def get_embeddings(self, texts: list[str], model: str = "qwen3-embedding:0.6b") -> list[list[float]]:
        if not texts:
            return []
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{self.ollama_url}/api/embed",
                    json={"model": model, "input": texts, "keep_alive": "30m"},
                )
            if resp.status_code == 200:
                return list(resp.json().get("embeddings", []))
        except Exception as exc:
            logger.warning("Failed to fetch Ollama embeddings (%s): %s", model, exc)
        return []

    async def abort_generation(self) -> bool:
        if self._is_ollama_backend() or self.get_backend() == "mlx":
            # Ollama / MLX cancels generation when the streaming HTTP connection is closed
            return True
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(f"{self.base_url}/api/extra/abort")
            return resp.status_code == 200
        except Exception as exc:
            logger.warning("Failed to call KoboldCpp abort: %s", exc)
            return False

    def _build_generate_payload(
        self,
        prompt: str,
        sampling_params: dict[str, Any],
        stop_sequences: list[str],
        max_context: int,
        seed: Optional[int] = None,
        temp_override: Optional[float] = None,
    ) -> dict[str, Any]:
        temp = temp_override if temp_override is not None else float(sampling_params.get("temp", 0.9))
        payload: dict[str, Any] = {
            "prompt": prompt,
            "use_story": False,
            "use_memory": False,
            "use_authors_note": False,
            "use_world_info": False,
            "max_context_length": int(max_context),
            "max_length": int(sampling_params.get("max_length", 744)),
            "temperature": temp,
            "min_p": float(sampling_params.get("min_p", 0.1)),
            "top_p": float(sampling_params.get("top_p", 1.0)),
            "top_k": int(sampling_params.get("top_k", 0)),
            "rep_pen": float(sampling_params.get("rep_pen", 1.04)),
            "rep_pen_range": int(sampling_params.get("rep_pen_range", 1024)),
            "dry_multiplier": float(sampling_params.get("dry_multiplier", 0.0)),
            "dry_base": float(sampling_params.get("dry_base", 1.75)),
            "dry_allowed_length": int(sampling_params.get("dry_allowed_length", 2)),
            "dry_sequence_breakers": list(sampling_params.get("dry_sequence_breakers", ["\n", ":", "\"", "*"])),
            "stop_sequence": stop_sequences,
        }
        if seed is not None:
            payload["sampler_seed"] = int(seed)
        return payload

    def _build_ollama_payload(
        self,
        ollama_model: str,
        prompt: str,
        sampling_params: dict[str, Any],
        stop_sequences: list[str],
        max_context: int,
        seed: Optional[int] = None,
        temp_override: Optional[float] = None,
    ) -> dict[str, Any]:
        temp = temp_override if temp_override is not None else float(sampling_params.get("temp", 0.9))
        rep_pen = float(sampling_params.get("rep_pen", 1.04))
        dry_mult = float(sampling_params.get("dry_multiplier", 0.0))
        # Ollama lacks native DRY sampler; boost repeat/presence/frequency penalties and use RP-tuned sampling
        effective_rep_pen = max(rep_pen, 1.15) if (dry_mult > 0 or rep_pen < 1.1) else rep_pen
        top_k = int(sampling_params.get("top_k", 0)) or 40
        top_p = min(float(sampling_params.get("top_p", 0.95)), 0.95)
        min_p = float(sampling_params.get("min_p", 0.06))
        options: dict[str, Any] = {
            "num_ctx": int(max_context or 65536),
            "num_predict": int(sampling_params.get("max_length", 8192)),
            "temperature": temp,
            "min_p": min_p,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": effective_rep_pen,
            "repeat_last_n": int(sampling_params.get("rep_pen_range", 2048)),
            "presence_penalty": float(sampling_params.get("presence_penalty", 0.45)),
            "frequency_penalty": float(sampling_params.get("frequency_penalty", 0.40)),
            "stop": stop_sequences,
        }
        if "mirostat" in sampling_params:
            options["mirostat"] = int(sampling_params["mirostat"])
            options["mirostat_tau"] = float(sampling_params.get("mirostat_tau", 5.0))
            options["mirostat_eta"] = float(sampling_params.get("mirostat_eta", 0.1))
        if seed is not None:
            options["seed"] = int(seed)
        return {
            "model": ollama_model,
            "prompt": prompt,
            "raw": True,
            "stream": True,
            "keep_alive": "15m",
            "options": options,
        }

    async def generate_stream(
        self,
        prompt: str,
        sampling_params: dict[str, Any],
        stop_sequences: list[str],
        max_context: int = 8192,
        cancel_event: Optional[asyncio.Event] = None,
        seed: Optional[int] = None,
        temp_override: Optional[float] = None,
    ) -> AsyncIterator[str]:
        self.touch_activity()
        timeout = httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0)

        async with self._gen_lock:
            self.last_prompt_eval_count = None
            self.last_eval_count = None
            if self.get_backend() == "mlx":
                cfg = self.get_model_info()
                mlx_model = str(cfg.get("mlx_model") or cfg.get("path") or "")
                temp = temp_override if temp_override is not None else float(sampling_params.get("temp", 0.9))
                mlx_payload: dict[str, Any] = {
                    "model": mlx_model,
                    "prompt": prompt,
                    "max_tokens": int(sampling_params.get("max_length", 1024)),
                    "temperature": temp,
                    "top_p": float(sampling_params.get("top_p", 1.0)),
                    "top_k": int(sampling_params.get("top_k", 0)),
                    "min_p": float(sampling_params.get("min_p", 0.1)),
                    "repetition_penalty": max(1.0, float(sampling_params.get("rep_pen", 1.08))),
                    "stop": stop_sequences,
                    "stream": True,
                }
                if seed is not None:
                    mlx_payload["seed"] = int(seed)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST",
                        f"{self.mlx_url}/v1/completions",
                        json=mlx_payload,
                    ) as resp:
                        resp.raise_for_status()
                        async for raw_line in resp.aiter_lines():
                            if cancel_event and cancel_event.is_set():
                                break
                            line = raw_line.strip()
                            if not line or not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if not data_str or data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            choices = chunk.get("choices") or []
                            if choices and isinstance(choices[0], dict):
                                token = choices[0].get("text")
                                if token:
                                    yield str(token)
                            usage = chunk.get("usage")
                            if isinstance(usage, dict):
                                if "prompt_tokens" in usage:
                                    self.last_prompt_eval_count = int(usage["prompt_tokens"])
                                if "completion_tokens" in usage:
                                    self.last_eval_count = int(usage["completion_tokens"])
                return

            if self._is_ollama_backend():
                if self._process is not None:
                    await self.stop_process()
                cfg = self.get_model_info()
                ollama_model = str(cfg.get("ollama_model", ""))
                if self.get_backend() == "ollama_cloud" and self.ollama_api_key:
                    ollama_model = str(cfg.get("ollama_cloud_model") or ollama_model.removesuffix("-cloud").removesuffix(":cloud"))
                endpoint_url, headers = self._ollama_endpoint_and_headers()
                ollama_payload = self._build_ollama_payload(
                    ollama_model=ollama_model,
                    prompt=prompt,
                    sampling_params=sampling_params,
                    stop_sequences=stop_sequences,
                    max_context=max_context,
                    seed=seed,
                    temp_override=temp_override,
                )
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST",
                        f"{endpoint_url}/api/generate",
                        headers=headers,
                        json=ollama_payload,
                    ) as resp:
                        resp.raise_for_status()
                        async for raw_line in resp.aiter_lines():
                            if cancel_event and cancel_event.is_set():
                                break
                            line = raw_line.strip()
                            if not line:
                                continue
                            try:
                                chunk = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            token = chunk.get("response")
                            if token:
                                yield str(token)
                            if chunk.get("done"):
                                if "prompt_eval_count" in chunk:
                                    self.last_prompt_eval_count = int(chunk["prompt_eval_count"])
                                if "eval_count" in chunk:
                                    self.last_eval_count = int(chunk["eval_count"])
                                break
                return

            payload = self._build_generate_payload(
                prompt=prompt,
                sampling_params=sampling_params,
                stop_sequences=stop_sequences,
                max_context=max_context,
                seed=seed,
                temp_override=temp_override,
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
                try:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/api/extra/generate/stream",
                        json=payload,
                    ) as resp:
                        if resp.status_code in (404, 405, 501):
                            raise NotImplementedError("SSE stream endpoint unavailable")
                        resp.raise_for_status()
                        async for raw_line in resp.aiter_lines():
                            if cancel_event and cancel_event.is_set():
                                await self.abort_generation()
                                break
                            line = raw_line.strip()
                            if not line or not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if not data_str or data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            token = chunk.get("token")
                            if token:
                                yield str(token)
                            if chunk.get("finish_reason") not in (None, "", "none"):
                                break
                        return
                except NotImplementedError:
                    pass

                resp = await client.post(f"{self.base_url}/api/v1/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results") or []
                if results and isinstance(results[0], dict):
                    text = results[0].get("text", "")
                    if text:
                        yield str(text)
