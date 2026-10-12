"""Atributos numéricos (estilo EMG) do laboratório, corrigidos e adaptados à base EEG-RAB2.

Origem: `utils/atributos_numericos/CalcAtributos.py` (código legado, mantido intacto). Aqui os 28 atributos são
calculados de forma vetorizada (janelas x canais), com fs = 300 Hz e os mesmos nomes. Correções em relação ao legado:

| Atributo | Legado | Corrigido |
|---|---|---|
| AAC, MAV, RMS, TM3-5, MVAL | divide por len(P) (n/2+1) | divide por N = nº de amostras da janela |
| SSC, ZCS | len(np.nonzero(...)) -> sempre 1 | np.count_nonzero (nº de mudanças de inclinação / cruzamentos de zero) |
| IAV | sum(k) | sum(|k|) |
| MLOGK | |mean(k)| | mean(log(|k| + eps)) |
| LOGD | exp(mean(log10(...))) (base misturada) | exp(mean(log(|k| + eps))) |
| periodograma | sem fs (frequência normalizada 0-0,5) | fs = 300 -> frequências em Hz |
| PKF | potência máxima | frequência do pico do espectro |
| MDF | metade da potência total | frequência mediana (potência acumulada = 50%) |
| VCF | ((SM2/TTP) - (SM1/TTP))^2 | SM2/TTP - (SM1/TTP)^2 (variância da frequência) |
| MAX | max(k) | max(|k|) |

Atributos positivos dependentes de escala e de cauda pesada entram em log10 (ver LOG10), para não dominarem a PCA.
Nomes das colunas: 'num_<ATRIBUTO>__<canal>' (prefixo 'num_' separa das famílias clássicas, ex.: num_KURT x kurtosis).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal, stats

from .data_io import FS

NAMES = ["AAC", "DASDV", "IAV", "LOGD", "MAV", "MLOGK", "RMS", "KURT", "SSC", "SSI", "VAR", "ZCS", "TM3", "TM4",
         "TM5", "STD", "MVAL", "MAX", "PSR", "MNF", "MNP", "PKF", "TTP", "SM1", "SM2", "SM3", "VCF", "MDF"]
LOG10 = {"AAC", "DASDV", "IAV", "LOGD", "MAV", "RMS", "SSI", "VAR", "TM3", "TM4", "TM5", "STD", "MAX", "MNP",
         "TTP", "SM1", "SM2", "SM3", "VCF"}
EPS = 1e-12


def numeric_attributes(windows: np.ndarray, fs: int = FS) -> dict[str, np.ndarray]:
    """windows: (n_janelas, n_amostras, n_canais) -> {atributo: (n_janelas, n_canais)} (antes do log10)."""
    x = windows.astype(np.float64)
    n = x.shape[1]
    d = np.diff(x, axis=1)
    ax = np.abs(x)
    logabs = np.log(ax + 1e-4)
    f, p = signal.periodogram(x, fs=fs, axis=1)
    ttp = p.sum(axis=1)
    sm1 = (f[None, :, None] * p).sum(axis=1)
    sm2 = (f[None, :, None] ** 2 * p).sum(axis=1)
    sm3 = (f[None, :, None] ** 3 * p).sum(axis=1)
    mnf = sm1 / (ttp + EPS)
    cum = np.cumsum(p, axis=1) / (ttp[:, None, :] + EPS)
    return {
        "AAC": np.abs(d).sum(axis=1) / n,
        "DASDV": np.sqrt((d ** 2).sum(axis=1) / (n - 1)),
        "IAV": ax.sum(axis=1),
        "LOGD": np.exp(logabs.mean(axis=1)),
        "MAV": ax.mean(axis=1),
        "MLOGK": logabs.mean(axis=1),
        "RMS": np.sqrt((x ** 2).mean(axis=1)),
        "KURT": stats.kurtosis(x, axis=1),
        "SSC": np.count_nonzero(np.diff(np.sign(d), axis=1), axis=1).astype(float),
        "SSI": (x ** 2).sum(axis=1),
        "VAR": x.var(axis=1),
        "ZCS": np.count_nonzero(np.diff(np.sign(x), axis=1), axis=1).astype(float),
        "TM3": np.abs((x ** 3).mean(axis=1)),
        "TM4": (x ** 4).mean(axis=1),
        "TM5": np.abs((x ** 5).mean(axis=1)),
        "STD": x.std(axis=1),
        "MVAL": x.mean(axis=1),
        "MAX": ax.max(axis=1),
        "PSR": p.max(axis=1) / (ttp + EPS),
        "MNF": mnf,
        "MNP": ttp / len(f),
        "PKF": f[p.argmax(axis=1)],
        "TTP": ttp,
        "SM1": sm1,
        "SM2": sm2,
        "SM3": sm3,
        "VCF": np.maximum(sm2 / (ttp + EPS) - mnf ** 2, 0.0),
        "MDF": f[(cum >= 0.5).argmax(axis=1)],
    }


def window_numeric_features(windows: np.ndarray, channels: list[str]) -> pd.DataFrame:
    """DataFrame (n_janelas, 28 x n_canais) com colunas 'num_<ATRIBUTO>__<canal>'."""
    feats = numeric_attributes(windows)
    cols = {}
    for name in NAMES:
        arr = feats[name]
        if name in LOG10:
            arr = np.log10(arr + EPS)
        for i, ch in enumerate(channels):
            cols[f"num_{name}__{ch}"] = arr[:, i].astype(np.float32)
    return pd.DataFrame(cols)
