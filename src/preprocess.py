"""Pré-processamento básico do EEG (scipy puro, sem MNE)."""
from __future__ import annotations

import numpy as np
from scipy import signal

from .data_io import FS


def bandpass(x: np.ndarray, low: float = 1.0, high: float = 45.0, fs: int = FS, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0).astype(np.float32)


def notch(x: np.ndarray, freq: float = 60.0, q: float = 30.0, fs: int = FS) -> np.ndarray:
    b, a = signal.iirnotch(freq, q, fs=fs)
    return signal.filtfilt(b, a, x, axis=0).astype(np.float32)


def common_average_reference(x: np.ndarray, good: np.ndarray | None = None) -> np.ndarray:
    good = np.ones(x.shape[1], bool) if good is None else good
    return (x - x[:, good].mean(axis=1, keepdims=True)).astype(np.float32)


def bad_channels(x: np.ndarray, flat_std: float = 0.5, z_thresh: float = 4.0) -> np.ndarray:
    """Canais planos (std < flat_std µV) ou com desvio-padrão robusto muito acima dos demais."""
    std = x.std(axis=0)
    med = np.median(std)
    mad = np.median(np.abs(std - med)) + 1e-9
    z = (std - med) / (1.4826 * mad)
    return (std < flat_std) | (z > z_thresh)


def amplitude_scale(x: np.ndarray, good: np.ndarray | None = None) -> float:
    """Escala robusta do registro: mediana (entre canais bons) do desvio robusto (1,4826·MAD) em µV.

    Um único fator por participante corrige diferenças de ganho/impedância entre sessões sem apagar a topografia
    nem o espectro relativo (escalar canal a canal apagaria diferenças espaciais reais)."""
    good = np.ones(x.shape[1], bool) if good is None else good
    xc = x[:, good] - np.median(x[:, good], axis=0)
    return float(np.median(1.4826 * np.median(np.abs(xc), axis=0)))


def preprocess(x: np.ndarray, car: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Notch 60 Hz -> passa-banda 1-45 Hz -> (opcional) referência média. Retorna (sinal, máscara de canais ruins)."""
    y = bandpass(notch(x))
    bad = bad_channels(y)
    if car:
        y = common_average_reference(y, ~bad)
    return y, bad
