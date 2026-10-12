"""Tratamento de artefatos: rejeição de janelas com músculo e ICA com classificação heurística de componentes.

Sem a ordem dos canais não dá para usar ICLabel (exige posições), então os componentes são classificados só por
propriedades temporais/espectrais. Os limiares são conservadores: na dúvida, o componente fica.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal, stats

from .data_io import FS

# Acima de ~70 Hz o equipamento atenua o sinal e em 125-148 Hz ele já encosta no piso de quantização
# (passo ~0,097 µV). 65-90 Hz acompanha a potência de 30-45 Hz janela a janela (Spearman ~0,94) e fica
# acima da gama cerebral e longe de 60/120 Hz -> faixa usada como marcador de EMG (fala, mandíbula).
MUSCLE_BAND = (65.0, 90.0)


def muscle_scores(windows: np.ndarray, band: tuple[float, float] = MUSCLE_BAND, fs: int = FS) -> np.ndarray:
    """log10 da potência na faixa muscular por janela (mediana entre canais). windows: (n_jan, n_amostras, n_canais)
    do sinal SEM passa-baixa (só notch)."""
    f, p = signal.welch(windows, fs=fs, nperseg=fs, axis=1)
    m = (f >= band[0]) & (f < band[1])
    return np.median(np.log10(p[:, m].mean(axis=1) + 1e-12), axis=1)


def reject_muscle(scores: np.ndarray, z_thresh: float = 3.0) -> np.ndarray:
    """Máscara de janelas aceitas: z robusto (dentro do participante) do escore muscular abaixo do limiar."""
    med = np.median(scores)
    mad = 1.4826 * np.median(np.abs(scores - med)) + 1e-9
    return (scores - med) / mad < z_thresh


# ---------------------------------------------------------------------------------------------- ICA

def _component_stats(s: np.ndarray, fs: int = FS) -> dict:
    """Propriedades de um componente (fonte ICA) usadas na classificação."""
    f, p = signal.welch(s, fs=fs, nperseg=4 * fs)
    tot = (f >= 1) & (f <= 45)
    low = (f >= 1) & (f < 4)
    fit = (f >= 7) & (f <= 45)
    slope = np.polyfit(np.log10(f[fit]), np.log10(p[fit] + 1e-30), 1)[0]
    # periodicidade tipo ECG: pico de autocorrelação em 0,5-1,5 s (40-120 bpm), após filtrar 5-30 Hz e retificar
    sos = signal.butter(4, [5, 30], btype="bandpass", fs=fs, output="sos")
    env = np.abs(signal.sosfiltfilt(sos, s[: min(len(s), 120 * fs)]))
    env = env - env.mean()
    ac = signal.correlate(env, env, mode="full", method="fft")[len(env) - 1:]
    ac = ac / (ac[0] + 1e-30)
    return {
        "low_ratio": p[low].sum() / (p[tot].sum() + 1e-30),
        "kurtosis": float(stats.kurtosis(s)),
        "slope_7_45": float(slope),
        "periodicity": float(ac[int(0.5 * fs): int(1.5 * fs)].max()),
    }


def classify_components(sources: np.ndarray, max_frac: float = 1 / 3) -> pd.DataFrame:
    """Classifica fontes (n_comp, n_amostras) em ocular / transitorio / muscular / cardiaco / cerebral.

    Limiares calibrados na distribuição de 760 componentes (38 participantes × 20) do v0
    (`data/processed/ica_component_stats_v0.csv`): curtose mediana 9,4 e p75 29 — base muito carregada de
    transitórios; componentes cerebrais são aproximadamente gaussianos.
      - ocular:      >= 85% da potência 1-45 Hz abaixo de 4 Hz E curtose > 20 (piscada: lenta e transitória)
      - transitorio: curtose > 50 (movimento/piscada de espectro largo; ~15% dos componentes)
    Com esses limiares a mediana é 4 componentes removidos por participante (teto atua em 8/38). Limiares mais
    frouxos (potência lenta >= 75%, curtose > 10) marcavam ~7 por participante — delta domina nesta base.
      - muscular:    inclinação do espectro 7-45 Hz (log-log) > -0,5 (EEG cai ~1/f; EMG é plano ou sobe)
      - cardiaco:    autocorrelação do envelope com pico > 0,4 entre 0,5-1,5 s E curtose > 3
    No máximo `max_frac` dos componentes são removidos (os de propriedade mais extrema têm prioridade).
    """
    df = pd.DataFrame([_component_stats(s) for s in sources])
    df["label"] = "cerebral"
    df.loc[(df["low_ratio"] >= 0.85) & (df["kurtosis"] > 20), "label"] = "ocular"
    df.loc[(df["kurtosis"] > 50) & (df["label"] == "cerebral"), "label"] = "transitorio"
    df.loc[(df["slope_7_45"] > -0.5) & (df["label"] == "cerebral"), "label"] = "muscular"
    df.loc[(df["periodicity"] > 0.4) & (df["kurtosis"] > 3) & (df["label"] == "cerebral"), "label"] = "cardiaco"
    flagged = df.index[df["label"] != "cerebral"]
    limit = int(np.floor(max_frac * len(df)))
    if len(flagged) > limit:  # mantém os mais "certos": propriedade mais extrema entre todos os componentes
        sev = df[["kurtosis", "slope_7_45", "periodicity"]].rank(pct=True).max(axis=1).loc[flagged]
        keep_flag = sev.sort_values(ascending=False).index[:limit]
        df.loc[flagged.difference(keep_flag), "label"] = "cerebral"
    df["exclude"] = df["label"] != "cerebral"
    return df


def run_ica(y: np.ndarray, n_components: int | None = None, seed: int = 0, fs: int = FS) -> tuple[np.ndarray, pd.DataFrame]:
    """ICA (FastICA via MNE) num sinal já filtrado 1-45 Hz e com referência média; classificação heurística.
    Com CAR o posto é n_canais - 1 (18 com os 19 canais de escalpo). Retorna (sinal limpo, tabela de componentes)."""
    import mne

    mne.set_log_level("ERROR")
    n_components = y.shape[1] - 1 if n_components is None else n_components
    info = mne.create_info([f"c{i:02d}" for i in range(1, y.shape[1] + 1)], fs, ch_types="eeg")
    raw = mne.io.RawArray(y.T.astype(np.float64) * 1e-6, info)  # MNE trabalha em volts
    ica = mne.preprocessing.ICA(n_components=n_components, method="fastica", max_iter="auto", random_state=seed)
    ica.fit(raw)
    comps = classify_components(ica.get_sources(raw).get_data())
    ica.exclude = comps.index[comps.exclude].tolist()
    clean = ica.apply(raw.copy()).get_data().T * 1e6
    return clean.astype(np.float32), comps


# ICLabel: rótulos possíveis e quais são removidos (com probabilidade >= ICLABEL_MIN_PROBA)
ICLABEL_REMOVE = ("eye blink", "muscle artifact", "heart beat", "line noise", "channel noise")
ICLABEL_MIN_PROBA = 0.7


def run_ica_iclabel(xn: np.ndarray, names: list[str], seed: int = 0, fs: int = FS) -> tuple[np.ndarray, pd.DataFrame]:
    """ICA (infomax estendida) + classificação automática com ICLabel (requer posições dos eletrodos).

    xn: sinal só com notch (n_amostras, n_canais), canais `names` (10-20). Seguindo o ICLabel, a ICA é ajustada numa
    cópia filtrada 1-100 Hz com referência média (o equipamento atenua acima de ~70 Hz, então a faixa alta é fraca
    nesta base). A mesma matriz de separação é aplicada ao sinal 1-45 Hz com referência média, que segue no pipeline.
    Remove componentes rotulados olho/músculo/coração/ruído de linha/ruído de canal com probabilidade >= 0,7.
    Retorna (sinal 1-45 Hz limpo, tabela com rótulo, probabilidade e exclusão por componente).
    """
    import mne
    from mne_icalabel import label_components

    from .data_io import mne_info
    from .preprocess import bandpass

    mne.set_log_level("ERROR")
    info = mne_info(names, fs)
    fit_raw = mne.io.RawArray(bandpass(xn, 1.0, 100.0).T.astype(np.float64) * 1e-6, info)
    fit_raw.set_eeg_reference("average")
    ica = mne.preprocessing.ICA(n_components=len(names) - 1, method="infomax", fit_params=dict(extended=True),
                                max_iter="auto", random_state=seed)
    ica.fit(fit_raw)
    out = label_components(fit_raw, ica, method="iclabel")
    comps = pd.DataFrame({"label": out["labels"], "proba": out["y_pred_proba"]})
    comps["exclude"] = comps["label"].isin(ICLABEL_REMOVE) & (comps["proba"] >= ICLABEL_MIN_PROBA)
    ica.exclude = comps.index[comps["exclude"]].tolist()
    apply_raw = mne.io.RawArray(bandpass(xn).T.astype(np.float64) * 1e-6, info)
    apply_raw.set_eeg_reference("average")
    clean = ica.apply(apply_raw).get_data().T * 1e6
    return clean.astype(np.float32), comps
