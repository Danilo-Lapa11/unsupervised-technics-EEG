"""Features clássicas de EEG por janela e por canal."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal, stats

from .data_io import CHANNELS, FS

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}


def band_powers(windows: np.ndarray, fs: int = FS) -> dict[str, np.ndarray]:
    """Potência absoluta e relativa por banda. windows: (n_jan, n_amostras, n_canais) -> arrays (n_jan, n_canais)."""
    freqs, psd = signal.welch(windows, fs=fs, nperseg=min(2 * fs, windows.shape[1]), axis=1)
    total_mask = (freqs >= 1) & (freqs <= 45)
    total = np.trapezoid(psd[:, total_mask], freqs[total_mask], axis=1)
    out = {}
    for name, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        p = np.trapezoid(psd[:, m], freqs[m], axis=1)
        out[f"abs_{name}"] = np.log10(p + 1e-12)
        out[f"rel_{name}"] = p / (total + 1e-12)
    out["ratio_theta_alpha"] = out["rel_theta"] / (out["rel_alpha"] + 1e-12)
    out["ratio_theta_beta"] = out["rel_theta"] / (out["rel_beta"] + 1e-12)
    pn = psd[:, total_mask] / (psd[:, total_mask].sum(axis=1, keepdims=True) + 1e-12)
    out["spectral_entropy"] = -(pn * np.log2(pn + 1e-12)).sum(axis=1) / np.log2(pn.shape[1])
    # frequência de pico na faixa 6-14 Hz (pico alfa; lentificação em TCM tende a deslocá-lo para baixo)
    am = (freqs >= 6) & (freqs <= 14)
    out["peak_alpha_freq"] = freqs[am][psd[:, am].argmax(axis=1)]
    return out


def hjorth(windows: np.ndarray) -> dict[str, np.ndarray]:
    d1 = np.diff(windows, axis=1)
    d2 = np.diff(d1, axis=1)
    v0, v1, v2 = windows.var(axis=1), d1.var(axis=1), d2.var(axis=1)
    mobility = np.sqrt(v1 / (v0 + 1e-12))
    complexity = np.sqrt(v2 / (v1 + 1e-12)) / (mobility + 1e-12)
    return {"hjorth_activity": np.log10(v0 + 1e-12), "hjorth_mobility": mobility, "hjorth_complexity": complexity}


def time_stats(windows: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "std": windows.std(axis=1),
        "skew": stats.skew(windows, axis=1),
        "kurtosis": stats.kurtosis(windows, axis=1),
    }


def window_features(windows: np.ndarray, channels: list[str] = CHANNELS) -> pd.DataFrame:
    """DataFrame (n_janelas, n_features*n_canais) com colunas '<feature>__<canal>'."""
    feats = {**band_powers(windows), **hjorth(windows), **time_stats(windows)}
    cols = {f"{name}__{ch}": arr[:, i] for name, arr in feats.items() for i, ch in enumerate(channels)}
    return pd.DataFrame(cols)


def feature_families(columns) -> pd.Series:
    return pd.Series([c.split("__")[0] for c in columns], index=columns)


def _lag1_autocorr(z: np.ndarray) -> np.ndarray:
    """Autocorrelação lag-1 por coluna de uma série já z-normalizada (n_janelas, n_features)."""
    if len(z) < 3:
        return np.full(z.shape[1], np.nan)
    return np.nanmean(z[1:] * z[:-1], axis=0) * len(z) / (len(z) - 1)


def subject_distribution_features(feat_w: pd.DataFrame, feats: list[str], max_windows: int = 40,
                                  by_family: bool = True) -> pd.DataFrame:
    """Descreve cada participante pela distribuição das suas janelas, em vez da média.

    Janelas são z-normalizadas dentro do participante (remove nível/escala individuais), logo média=0 e std=1 para
    todos; o que sobra é a *forma* (p10, p90, assimetria, curtose) e a *dinâmica* (autocorrelação lag-1 entre janelas
    consecutivas). Soma-se a variabilidade bruta (IQR, sem z-score). Usa as primeiras `max_windows` janelas retidas
    (bloco contíguo) para que o nº de janelas — confundido com o grupo — não enviese as estimativas.
    Com `by_family`, cada estatística é a média entre canais da família de feature (reduz dimensão para n pequeno).
    """
    fam = feature_families(feats)
    rows = []
    for subj, d in feat_w.sort_values("window").groupby("subject", sort=False):
        raw = d[feats].iloc[:max_windows]
        z = ((raw - raw.mean()) / (raw.std() + 1e-9)).to_numpy()
        per_feat = {
            "p10": np.nanpercentile(z, 10, axis=0),
            "p90": np.nanpercentile(z, 90, axis=0),
            "skew": stats.skew(z, axis=0, nan_policy="omit"),
            "kurt": stats.kurtosis(z, axis=0, nan_policy="omit"),
            "acf1": _lag1_autocorr(z),
            "iqr_raw": (raw.quantile(0.75) - raw.quantile(0.25)).to_numpy(),
        }
        row = {"subject": subj, "group": d["group"].iloc[0], "sex": d["sex"].iloc[0], "n_windows_used": len(raw)}
        for stat, vals in per_feat.items():
            s = pd.Series(np.asarray(vals, float), index=feats)
            if by_family:
                s = s.groupby(fam).mean()
            row |= {f"{stat}__{k}": v for k, v in s.items()}
        rows.append(row)
    return pd.DataFrame(rows)
