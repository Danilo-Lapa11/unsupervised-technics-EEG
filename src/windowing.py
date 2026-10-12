"""Janelamento do sinal (padrão da tese: 5 s com 1 s de sobreposição)."""
from __future__ import annotations

import numpy as np

from .data_io import FS


def sliding_windows(x: np.ndarray, win_s: float = 5.0, overlap_s: float = 1.0, fs: int = FS) -> np.ndarray:
    """(n_amostras, n_canais) -> (n_janelas, win, n_canais). Descarta o resto final."""
    win = int(round(win_s * fs))
    step = win - int(round(overlap_s * fs))
    if x.shape[0] < win:
        return np.empty((0, win, x.shape[1]), dtype=x.dtype)
    starts = np.arange(0, x.shape[0] - win + 1, step)
    return np.stack([x[s:s + win] for s in starts])


def reject_windows(windows: np.ndarray, z_thresh: float = 3.0, max_abs_uv: float = 500.0) -> np.ndarray:
    """Máscara de janelas aceitas.

    Amplitude da janela = mediana (entre canais) do pico absoluto, robusta a um único canal ruim. Rejeita janelas
    atípicas para o próprio participante (z robusto > z_thresh) ou acima de um teto absoluto. Um limiar fixo em
    todos os canais (ex.: 150 µV) descartava quase todas as janelas de vários sujeitos, pois o relato envolve fala.
    """
    amp = np.median(np.abs(windows).max(axis=1), axis=1)
    med = np.median(amp)
    mad = 1.4826 * np.median(np.abs(amp - med)) + 1e-9
    return ((amp - med) / mad < z_thresh) & (amp < max_abs_uv)
