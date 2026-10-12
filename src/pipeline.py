"""Pipeline de pré-processamento + features configurável, para comparar variantes com o baseline v0.

Ordem: bruto -> [só 19 canais de escalpo] -> notch 60 (+120) Hz -> [escore muscular por janela] -> passa-banda 1-45 Hz
       -> canais ruins -> CAR -> [ICA heurística | ICA + ICLabel] -> janelas 5 s/1 s -> rejeição por amplitude
       (+ muscular) -> [escala por participante] -> features clássicas [+ atributos numéricos]
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .data_io import PROCESSED_DIR, CHANNELS, SCALP_CHANNELS, load_metadata, load_subject, select_channels
from .preprocess import notch, bandpass, bad_channels, common_average_reference, amplitude_scale
from .windowing import sliding_windows, reject_windows
from .features import window_features
from .features_numericos import window_numeric_features
from .artifacts import muscle_scores, reject_muscle, run_ica, run_ica_iclabel


@dataclass(frozen=True)
class Config:
    name: str
    drop_ears: bool = True       # excluir A1/A2 (eletrodos de orelha): análises com os 19 canais de escalpo
    notch_120: bool = False      # 2ª harmônica da rede (necessário para olhar acima de 100 Hz)
    muscle: bool = False         # rejeitar janelas com EMG (potência 65-90 Hz atípica para o participante)
    ica: str | None = None       # None | "heuristica" | "iclabel"
    scale: bool = False          # dividir o sinal por um fator robusto de amplitude do participante
    numericos: bool = False      # calcular também os 28 atributos numéricos (src/features_numericos.py)
    exclude: tuple = ()          # canais de escalpo a remover antes de CAR/ICA (ex.: ("F4",), suspeito de problema técnico)
    muscle_z: float = 3.0


VARIANTS = {
    "v0": Config("v0", numericos=True),
    "v1_full": Config("v1_full", notch_120=True, muscle=True, ica="heuristica", scale=True),
    "v2_iclabel": Config("v2_iclabel", ica="iclabel"),
    "v2_full": Config("v2_full", ica="iclabel", scale=True),
    # v2_full sem F4 (atenuação de alta frequência em F4, mais frequente no TCM — ver achados, seção 11)
    "v2_full_semF4": Config("v2_full_semF4", ica="iclabel", scale=True, numericos=True, exclude=("F4",)),
    # histórico: baseline original com os 21 canais (inclui A1/A2), para reprodução
    "v0_21canais": Config("v0_21canais", drop_ears=False),
}


def channel_names(cfg: Config) -> list[str]:
    names = SCALP_CHANNELS if cfg.drop_ears else CHANNELS
    return [n for n in names if n not in cfg.exclude]


def _ica_cache(subject: str, method: str, exclude: tuple = ()):
    d = PROCESSED_DIR / ("ica" if method == "heuristica" else f"ica_{method}")
    if exclude:  # ICA com outro conjunto de canais -> cache separado
        d = d.with_name(d.name + "_sem" + "_".join(exclude))
    return d / f"{subject}.parquet", d / f"{subject}_components.csv"


def clean_signal(x: np.ndarray, cfg: Config, subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Retorna (sinal 1-45 Hz limpo, sinal só com notch [p/ escore muscular], máscara de canais ruins, log)."""
    log = {}
    names = channel_names(cfg)
    if names != CHANNELS:
        x = select_channels(x, names)
    xn = notch(x, 60.0)
    if cfg.notch_120:
        xn = notch(xn, 120.0)
    y = bandpass(xn)
    bad = bad_channels(y)
    y = common_average_reference(y, ~bad)
    log["n_bad_channels"] = int(bad.sum())
    log["bad_channels"] = " ".join(n for n, b in zip(names, bad) if b)
    if cfg.ica:
        sig_path, comp_path = _ica_cache(subject, cfg.ica, cfg.exclude)
        if sig_path.exists():  # ICA é a etapa cara: reaproveita entre variantes que a usam
            y = pd.read_parquet(sig_path).to_numpy(np.float32)
            comps = pd.read_csv(comp_path)
        else:
            y, comps = run_ica(y) if cfg.ica == "heuristica" else run_ica_iclabel(xn, names)
            sig_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(y, columns=names).to_parquet(sig_path)
            comps.to_csv(comp_path, index=False)
        log["ica_removidos"] = int(comps["exclude"].sum())
        for lab, n in comps.loc[comps["exclude"], "label"].value_counts().items():
            log[f"ica_{lab.replace(' ', '_')}"] = int(n)
    return y, xn, bad, log


def subject_features(subject: str, cfg: Config, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    names = channel_names(cfg)
    x = load_subject(subject, meta)
    y, xn, bad, log = clean_signal(x, cfg, subject)
    w = sliding_windows(y)
    keep = reject_windows(w)
    log["n_windows"] = len(w)
    log["rej_amplitude"] = int((~keep).sum())
    if cfg.muscle:
        m_ok = reject_muscle(muscle_scores(sliding_windows(xn)), cfg.muscle_z)
        log["rej_muscle_only"] = int((keep & ~m_ok).sum())
        keep &= m_ok
    log["kept"] = int(keep.sum())
    scale = amplitude_scale(y, ~bad) if cfg.scale else 1.0
    log["scale_uV"] = amplitude_scale(y, ~bad)  # sempre registrado (para checar confundidor de amplitude)
    wk = w[keep] / scale
    f = window_features(wk, channels=names)
    f.insert(0, "window", np.flatnonzero(keep))
    fn = None
    if cfg.numericos:  # mesma máscara de janelas -> junção por (subject, window) com as clássicas
        fn = window_numeric_features(wk, names)
        fn.insert(0, "window", np.flatnonzero(keep))
    return f, fn, log


def build_features(cfg: Config, meta: pd.DataFrame | None = None, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features por janela (todas as pessoas) + log por participante. Salva em data/processed/<nome>/."""
    meta = load_metadata() if meta is None else meta
    parts, parts_n, logs = [], [], []
    for r in meta.itertuples(index=False):
        f, fn, log = subject_features(r.subject, cfg, meta)
        for df in (f, fn):
            if df is not None:
                df.insert(0, "sex", r.sex); df.insert(0, "group", r.group); df.insert(0, "subject", r.subject)
        parts.append(f)
        if fn is not None:
            parts_n.append(fn)
        logs.append({"subject": r.subject, "group": r.group, **log})
        if verbose:
            print(f"[{cfg.name}] {r.subject}: {log}", flush=True)
    feat_w = pd.concat(parts, ignore_index=True)
    logs = pd.DataFrame(logs)
    out = PROCESSED_DIR / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    feat_w.to_parquet(out / "features_windows.parquet")
    if parts_n:
        pd.concat(parts_n, ignore_index=True).to_parquet(out / "features_numericos.parquet")
    logs.to_csv(out / "log.csv", index=False)
    pd.Series(asdict(cfg)).to_json(out / "config.json")
    return feat_w, logs


def load_variant(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = PROCESSED_DIR / name
    return pd.read_parquet(out / "features_windows.parquet"), pd.read_csv(out / "log.csv")


def load_numericos(name: str = "v0") -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / name / "features_numericos.parquet")
