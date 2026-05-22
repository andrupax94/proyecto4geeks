"""
realtime_monitor.py
────────────────────────────────────────────────────────────────────────────
Monitor de audio en tiempo real. Autoinvocable, sin argumentos.

Ejecutar:
    python realtime_monitor.py

Requiere:
    pip install sounddevice torch torchaudio numpy
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import shutil
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

# ─── rutas del proyecto ──────────────────────────────────────────────────────
# Ajusta este import si la estructura de tu proyecto difiere
try:
    from src.utils.config import (
        LABEL_MAPPING,
        FINAL_MODEL_DIR,
        CHECKPOINT_DIR,
    )
    # Selecciona el label mapping binario alertable
    DEFAULT_LABEL_MAPPING_PATH: Path = LABEL_MAPPING["alertable2"]
    # Busca el mejor checkpoint en FINAL_MODEL_DIR; si no existe prueba CHECKPOINT_DIR
    _model_candidates = sorted(FINAL_MODEL_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not _model_candidates:
        _model_candidates = sorted(CHECKPOINT_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    DEFAULT_MODEL_PATH: Path = _model_candidates[0] if _model_candidates else Path("model.pt")
except Exception as _cfg_err:
    # Fallback manual si el import falla
    print(f"[WARN] No se pudo importar src.utils.config ({_cfg_err}). "
          "Usando rutas por defecto.")
    DEFAULT_MODEL_PATH        = Path("model.pt")
    DEFAULT_LABEL_MAPPING_PATH = Path("label_mapping_alertable.pkl")

try:
    from src.models.hybrid_cnn_v3 import ImprovedMFCCCNN
except ImportError as _model_err:
    print(f"[ERROR] No se pudo importar ImprovedMFCCCNN: {_model_err}")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY — consola ANSI
# ═══════════════════════════════════════════════════════════════════════════════

W = shutil.get_terminal_size((80, 20)).columns

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RED    = "\033[91m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_WHITE  = "\033[97m"
_GREY   = "\033[90m"


def _bar(value: float, width: int = 28, filled: str = "█", empty: str = "░") -> str:
    filled_n = int(round(value * width))
    return filled * filled_n + empty * (width - filled_n)


def _color_for_prob(p: float) -> str:
    if p >= 0.75:
        return _RED
    if p >= 0.50:
        return _YELLOW
    return _GREEN


def clear_line() -> None:
    print("\033[2K\r", end="")


def print_header(model_path: Path, mode: str, chunk: float, step: float) -> None:
    line = "─" * W
    print(f"\n{_BOLD}{_CYAN}{'MONITOR DE AUDIO EN TIEMPO REAL':^{W}}{_RESET}")
    print(f"{_DIM}{line}{_RESET}")
    print(f"  {_GREY}Modelo :{_RESET} {model_path.name}")
    print(f"  {_GREY}Modo   :{_RESET} {mode}   "
          f"{_GREY}Ventana:{_RESET} {chunk:.1f}s   "
          f"{_GREY}Paso   :{_RESET} {step:.1f}s")
    print(f"{_DIM}{line}{_RESET}")
    print(f"  {_GREY}Ctrl+C para salir{_RESET}\n")


def print_tick(result: dict, tick: int) -> None:
    """Imprime una línea por iteración, borrando la anterior."""
    alert   = result["alert"]
    p       = result["p_alert"]
    smooth  = result["smooth"]
    reason  = result["reason"]
    top     = result["top_label"]
    top_p   = result["top_prob"]
    recent  = result["recent"]
    votes   = result.get("votes", "—")

    color   = _RED if alert else (_YELLOW if p >= 0.50 else _GREEN)
    icon    = "🔴 ALERTA" if alert else ("🟡 DUDOSO" if p >= 0.45 else "🟢 Normal")

    bar_color = _color_for_prob(p)
    bar = _bar(p)

    recent_str = " ".join(f"{v:.2f}" for v in recent[-5:])

    ts = time.strftime("%H:%M:%S")

    # Línea principal
    print(f"\r{_BOLD}{color}{icon:<14}{_RESET}"
          f"  {bar_color}{bar}{_RESET} {p:.3f}"
          f"  {_DIM}smooth={smooth:.3f}  top={top}({top_p:.2f})  "
          f"votos={votes}  [{recent_str}]  {ts}{_RESET}",
          end="\n" if alert else "\r",
          flush=True)

    if alert:
        banner = f"  ⚠  ALERTA DISPARADA — razón: {reason}  ⚠  "
        pad = max(0, W - len(banner))
        print(f"{_BOLD}{_RED}{banner}{' ' * pad}{_RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MonitorConfig:
    # Modelo / preproceso
    model_sample_rate: int   = 16_000
    target_duration:   float = 3.0
    n_fft:             int   = 1024
    hop_length:        int   = 160
    n_mels:            int   = 128
    n_mfcc:            int   = 13
    peak_target:       float = 0.99
    mode:              str   = "mel_waveform"   # mel_only | mel_mfcc | mel_waveform

    # Captura
    chunk_seconds:         float        = 3.0
    step_seconds:          float        = 1.0
    capture_sample_rate:   Optional[int] = None   # None → usa el del dispositivo
    channels:              int           = 1

    # TTA — perturbaciones ligeras sobre cada chunk
    tta_rounds:   int   = 4       # ≥2 para activar TTA
    tta_noise:    float = 0.002
    tta_gain_min: float = 0.90
    tta_gain_max: float = 1.10

    # ── Decisión (conservadora) ───────────────────────────────────────────────
    #
    # Filosofía: los no-alertables se clasifican bien → relajamos el "clear"
    # pero exigimos consenso fuerte antes de disparar una alerta real.
    #
    # Historial
    history_size: int = 8          # ventana deslizante de probabilidades
    #
    # Suavizado — media ponderada lineal (más peso a lo reciente)
    alert_threshold: float = 0.68  # promedio ponderado debe superar esto
    #
    # Votación — de las últimas N ventanas, cuántas deben estar sobre el umbral
    vote_window:   int   = 5       # últimas N para votar
    votes_needed:  int   = 4       # de esas N, mínimo necesario  (80 %)
    vote_min_conf: float = 0.72    # prob mínima para que cuente como voto
    #
    # Pico único de muy alta confianza (disparo inmediato)
    spike_threshold: float = 0.94  # solo dispara si es casi certero
    #
    # Clear — cuando el modelo está tranquilo
    clear_threshold: float = 0.30
    #
    # Cooldown entre alertas consecutivas
    cooldown_seconds: float = 6.0


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════════

def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def normalize_bool_label(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f"}:
        return False
    return None


def resolve_alert_index(label_mapping: dict) -> Tuple[int, Dict[int, Any]]:
    idx2label = label_mapping.get("idx2label", {})
    label2idx = label_mapping.get("label2idx", {})

    alert_index = None
    if True in label2idx:
        alert_index = int(label2idx[True])
    elif "true" in label2idx:
        alert_index = int(label2idx["true"])

    if alert_index is None:
        for idx, lab in idx2label.items():
            if normalize_bool_label(lab) is True:
                alert_index = int(idx)
                break

    if alert_index is None:
        if len(idx2label) == 2:
            alert_index = max(idx2label.keys())
        else:
            raise ValueError("No pude resolver el índice alertable desde el label mapping.")

    return alert_index, idx2label


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESO
# ═══════════════════════════════════════════════════════════════════════════════

class LivePreprocessor:
    def __init__(self, cfg: MonitorConfig, device: torch.device):
        self.cfg    = cfg
        self.device = device

        self.mel_transform = T.MelSpectrogram(
            sample_rate=cfg.model_sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            power=2.0,
        ).to(device)

        self.amplitude_to_db = T.AmplitudeToDB(stype="power").to(device)

        self.mfcc_transform = T.MFCC(
            sample_rate=cfg.model_sample_rate,
            n_mfcc=cfg.n_mfcc,
            melkwargs={
                "n_fft":       cfg.n_fft,
                "hop_length":  cfg.hop_length,
                "n_mels":      cfg.n_mels,
                "center":      True,
                "power":       2.0,
            },
        ).to(device)

    @staticmethod
    def _ensure_mono(w: torch.Tensor) -> torch.Tensor:
        if w.ndim == 1:
            w = w.unsqueeze(0)
        if w.shape[0] > 1:
            w = w.mean(dim=0, keepdim=True)
        return w

    def _normalize_peak(self, w: torch.Tensor) -> torch.Tensor:
        peak = w.abs().max().clamp_min(1e-8)
        return w / peak * self.cfg.peak_target

    def _fix_length(self, w: torch.Tensor) -> torch.Tensor:
        target = int(self.cfg.model_sample_rate * self.cfg.target_duration)
        n = w.shape[-1]
        if n > target:
            return w[..., :target]
        if n < target:
            return F.pad(w, (0, target - n))
        return w

    def _resample(self, w: torch.Tensor, orig_sr: int) -> torch.Tensor:
        if orig_sr == self.cfg.model_sample_rate:
            return w
        return T.Resample(orig_freq=orig_sr, new_freq=self.cfg.model_sample_rate).to(self.device)(w)

    def to_features(
        self, waveform_np: np.ndarray, orig_sr: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        w = torch.tensor(waveform_np, dtype=torch.float32, device=self.device)
        w = self._ensure_mono(w)
        w = self._resample(w, orig_sr)
        w = self._fix_length(w)
        w = self._normalize_peak(w)

        mode = self.cfg.mode
        with torch.inference_mode():
            mel  = self.amplitude_to_db(self.mel_transform(w)).float()
            mfcc = self.mfcc_transform(w).float() if mode == "mel_mfcc" else None

        if mode == "mel_only":
            return mel, None, None
        if mode == "mel_mfcc":
            return mel, mfcc, None
        # mel_waveform (default)
        return mel, None, w.float()


# ═══════════════════════════════════════════════════════════════════════════════
# BUFFER DE DECISIÓN (conservador)
# ═══════════════════════════════════════════════════════════════════════════════

class ConservativeDecisionBuffer:
    """
    Dispara una alerta solo cuando existe un claro consenso multi-ventana.
    Estrategia de tres vías independientes — cualquiera puede disparar:

      1. Promedio ponderado ≥ alert_threshold  (tendencia sostenida)
      2. Votación: de las últimas vote_window ventanas,
         al menos votes_needed superan vote_min_conf  (mayoría cualificada)
      3. Pico único ≥ spike_threshold  (certeza abrumadora)

    En todos los casos se respeta el cooldown.
    """

    def __init__(self, cfg: MonitorConfig):
        self.cfg = cfg
        self.probs: Deque[float] = deque(maxlen=cfg.history_size)
        self.last_alert_time: float = 0.0

    def add(self, p: float) -> dict:
        self.probs.append(float(p))
        recent = list(self.probs)

        # ── métricas ─────────────────────────────────────────────────────────
        weights = np.arange(1, len(recent) + 1, dtype=np.float32)
        smooth  = float(np.dot(np.array(recent, dtype=np.float32), weights) / weights.sum())

        last_n       = recent[-self.cfg.vote_window:]
        votes        = sum(v >= self.cfg.vote_min_conf for v in last_n)
        mean_p       = float(np.mean(recent))
        max_p        = max(recent)

        now        = time.time()
        cooldown_ok = (now - self.last_alert_time) >= self.cfg.cooldown_seconds

        # ── decisión ─────────────────────────────────────────────────────────
        alert  = False
        reason = "stable_no_alert"

        if cooldown_ok:
            if smooth >= self.cfg.alert_threshold:
                alert  = True
                reason = "smooth_threshold"
            elif votes >= self.cfg.votes_needed:
                alert  = True
                reason = f"vote_consensus ({votes}/{len(last_n)})"
            elif max_p >= self.cfg.spike_threshold:
                alert  = True
                reason = "single_spike_high_conf"

        if alert:
            self.last_alert_time = now

        if not alert and mean_p <= self.cfg.clear_threshold and max_p < self.cfg.vote_min_conf:
            reason = "clear"

        return {
            "alert":  alert,
            "reason": reason,
            "smooth": smooth,
            "mean":   mean_p,
            "max":    max_p,
            "votes":  f"{votes}/{len(last_n)}",
            "recent": recent,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODELO
# ═══════════════════════════════════════════════════════════════════════════════

class RealtimeModel:
    def __init__(
        self,
        model_path: Path,
        label_mapping_path: Path,
        device: torch.device,
        cfg: MonitorConfig,
    ):
        self.cfg    = cfg
        self.device = device

        mapping = load_pickle(label_mapping_path)
        self.alert_index, self.idx2label = resolve_alert_index(mapping)
        num_classes = int(mapping["num_classes"])

        self.model = ImprovedMFCCCNN(
            num_classes=num_classes,
            dropout=0.25,
            mode=cfg.mode,
        ).to(device)

        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        self.pre     = LivePreprocessor(cfg, device)
        self.buffer  = ConservativeDecisionBuffer(cfg)
        self.softmax = torch.nn.Softmax(dim=1)

    # ── inferencia ───────────────────────────────────────────────────────────

    def _forward(
        self,
        mel: torch.Tensor,
        mfcc: Optional[torch.Tensor],
        waveform: Optional[torch.Tensor],
    ) -> torch.Tensor:
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(mel=mel, mfcc=mfcc, waveform=waveform)
            else:
                logits = self.model(mel=mel, mfcc=mfcc, waveform=waveform)
            return self.softmax(logits).squeeze(0)

    def predict(self, waveform_np: np.ndarray, orig_sr: int) -> dict:
        mel, mfcc, wav = self.pre.to_features(waveform_np, orig_sr)

        # ── TTA ──────────────────────────────────────────────────────────────
        preds: list[torch.Tensor] = []
        for i in range(max(1, self.cfg.tta_rounds)):
            mel_i, mfcc_i, wav_i = mel, mfcc, wav

            if self.cfg.tta_rounds > 1 and wav_i is not None:
                gain   = torch.empty(1, device=self.device).uniform_(
                    self.cfg.tta_gain_min, self.cfg.tta_gain_max
                )
                wav_i  = wav_i * gain
                if self.cfg.tta_noise > 0:
                    wav_i = wav_i + torch.randn_like(wav_i) * self.cfg.tta_noise
                peak   = wav_i.abs().max().clamp_min(1e-8)
                wav_i  = wav_i / peak * self.cfg.peak_target

                # Recomputa mel con el waveform perturbado
                with torch.inference_mode():
                    mel_i = self.pre.amplitude_to_db(
                        self.pre.mel_transform(wav_i)
                    ).float()

            probs = self._forward(
                mel_i.unsqueeze(0),
                None if mfcc_i is None else mfcc_i.unsqueeze(0),
                None if wav_i  is None else wav_i.unsqueeze(0),
            )
            preds.append(probs)

        mean_probs = torch.stack(preds, dim=0).mean(0)
        p_alert    = float(mean_probs[self.alert_index].item())
        decision   = self.buffer.add(p_alert)

        top_idx   = int(torch.argmax(mean_probs).item())
        top_label = self.idx2label.get(top_idx, top_idx)
        top_prob  = float(mean_probs[top_idx].item())

        return {
            "p_alert":    p_alert,
            "top_label":  top_label,
            "top_prob":   top_prob,
            "class_probs": {
                str(self.idx2label.get(i, i)): float(mean_probs[i].item())
                for i in range(mean_probs.numel())
            },
            **decision,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SELECCIÓN DE DISPOSITIVO
# ═══════════════════════════════════════════════════════════════════════════════

def list_input_devices() -> list[dict]:
    devices = sd.query_devices()
    inputs  = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    print(f"\n{_BOLD}Dispositivos de entrada disponibles:{_RESET}")
    for idx, dev in inputs:
        sr = int(dev["default_samplerate"])
        default_tag = f"  {_YELLOW}← default{_RESET}" if idx == sd.default.device[0] else ""
        print(f"  {_CYAN}[{idx:2d}]{_RESET} {dev['name']:<42} "
              f"{_DIM}ch={dev['max_input_channels']}  sr={sr}{_RESET}{default_tag}")
    return inputs


def choose_device() -> Tuple[int, int]:
    """
    Devuelve (device_index, capture_samplerate).
    """
    inputs = list_input_devices()
    if not inputs:
        print(f"{_RED}No se encontraron dispositivos de entrada.{_RESET}")
        sys.exit(1)

    default_idx = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else 0
    raw = input(f"\n{_BOLD}Índice del dispositivo [{default_idx}]: {_RESET}").strip()
    dev_idx = int(raw) if raw else default_idx

    dev_info = sd.query_devices(dev_idx, "input")
    sr = int(dev_info["default_samplerate"])
    print(f"\n{_GREEN}✓ Usando: [{dev_idx}] {dev_info['name']}  |  SR={sr}{_RESET}\n")
    return dev_idx, sr


# ═══════════════════════════════════════════════════════════════════════════════
# BUCLE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def run() -> None:
    cfg = MonitorConfig()

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── selección de rutas ───────────────────────────────────────────────────
    model_path   = DEFAULT_MODEL_PATH
    mapping_path = DEFAULT_LABEL_MAPPING_PATH

    # Comprobación rápida
    for p, label in [(model_path, "Modelo"), (mapping_path, "Label mapping")]:
        if not Path(p).exists():
            print(f"{_RED}[ERROR] {label} no encontrado: {p}{_RESET}")
            print("Edita DEFAULT_MODEL_PATH / DEFAULT_LABEL_MAPPING_PATH al inicio del script.")
            sys.exit(1)

    # ── dispositivo de captura ───────────────────────────────────────────────
    dev_idx, capture_sr = choose_device()

    # ── cabecera ─────────────────────────────────────────────────────────────
    print_header(Path(model_path), cfg.mode, cfg.chunk_seconds, cfg.step_seconds)

    # ── carga del modelo ─────────────────────────────────────────────────────
    print(f"  {_DIM}Cargando modelo en {torch_device}...{_RESET}", end=" ", flush=True)
    model = RealtimeModel(
        model_path=Path(model_path),
        label_mapping_path=Path(mapping_path),
        device=torch_device,
        cfg=cfg,
    )
    print(f"{_GREEN}listo{_RESET}\n")

    frames = int(round(cfg.chunk_seconds * capture_sr))

    # ── bucle ────────────────────────────────────────────────────────────────
    tick = 0
    while True:
        try:
            t0    = time.time()
            audio = sd.rec(
                frames,
                samplerate=capture_sr,
                channels=cfg.channels,
                dtype="float32",
                device=dev_idx,
                blocking=True,
            )
            audio = np.asarray(audio, dtype=np.float32)

            result = model.predict(audio, orig_sr=capture_sr)
            print_tick(result, tick)
            tick += 1

            elapsed   = time.time() - t0
            sleep_for = max(0.0, cfg.step_seconds - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)

        except KeyboardInterrupt:
            print(f"\n\n{_DIM}Monitor detenido.{_RESET}\n")
            break
        except Exception as exc:
            print(f"\n{_YELLOW}[WARN] Error en iteración: {exc}{_RESET}")
            time.sleep(1.0)


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run()