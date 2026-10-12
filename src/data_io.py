"""Leitura da base EEG-RAB2 e metadados derivados do nome dos arquivos."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "EEG-RAB2"
PROCESSED_DIR = ROOT / "data" / "processed"
METADATA_PATH = ROOT / "data" / "metadata.csv"
AFFECT_PATH = ROOT / "data" / "rotulos_afetivos_rab2.csv"

FS = 300  # Hz (Neurovirtual BWIII, tese cap. 5.2)
N_CHANNELS = 21
# Ordem das colunas informada pela orientadora (03/10/2026). Nomenclatura 10-20 antiga: T3/T4 = T7/T8, T5/T6 = P7/P8.
# Ímpar = hemisfério esquerdo, par = direito, "z" = linha média. Validação pelos dados: correlação média entre canais
# x distância 3D (montagem standard_1020) dá Spearman -0,52 (mais forte que todas as 5.000 ordens aleatórias testadas;
# mínimo -0,31). A Figura 27 da tese destaca CP5/CP6 no lugar de T5/T6 — vale a lista da orientadora.
CHANNELS = ["Fp1", "Fp2", "F7", "F8", "F3", "F4", "T3", "T4", "C3", "C4", "T5", "T6",
            "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz", "A1", "A2"]
REGION = {
    **dict.fromkeys(["Fp1", "Fp2", "F7", "F8", "F3", "F4", "Fz"], "frontal"),
    **dict.fromkeys(["T3", "T4", "T5", "T6"], "temporal"),
    **dict.fromkeys(["C3", "C4", "Cz"], "central"),
    **dict.fromkeys(["P3", "P4", "Pz"], "parietal"),
    **dict.fromkeys(["O1", "O2"], "occipital"),
    **dict.fromkeys(["A1", "A2"], "auricular"),
}
HEMISPHERE = {ch: "linha_media" if ch.endswith("z") else ("esquerdo" if int(ch[-1]) % 2 else "direito")
              for ch in CHANNELS}
# A1/A2 são eletrodos de orelha (referência típica; captam ECG e músculo, não córtex) e eram os canais mais marcados
# como ruins -> por padrão as análises usam só os 19 canais de escalpo.
SCALP_CHANNELS = [ch for ch in CHANNELS if REGION[ch] != "auricular"]


def select_channels(x: np.ndarray, names: list[str] = SCALP_CHANNELS) -> np.ndarray:
    """Seleciona colunas do sinal (n_amostras, 21) pelos nomes, na ordem de `names`."""
    return x[:, [CHANNELS.index(n) for n in names]]


def mne_info(names: list[str] = SCALP_CHANNELS, fs: int = FS):
    """Info do MNE com montagem 10-20 (para topomaps e ICLabel)."""
    import warnings
    import mne
    info = mne.create_info(list(names), fs, ch_types="eeg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # 'standard_1020' será renomeada no MNE 1.14
        info.set_montage(mne.channels.make_standard_montage("standard_1020"))
    return info

_FILENAME_RE = re.compile(r"^P(?P<num>\d+)_(?P<group>CTR|TCM)_(?P<sex>[FM])_EEG_(?P<segment>\w+)\.csv$")


def parse_filename(name: str) -> dict:
    m = _FILENAME_RE.match(Path(name).name)
    if m is None:
        raise ValueError(f"Nome fora do padrão P#_GRUPO_SEXO_EEG_TRECHO.csv: {name}")
    num = int(m["num"])
    return {
        "subject": f"P{num}",
        "subject_num": num,
        "group": m["group"],
        "sex": m["sex"],
        "segment": m["segment"],
        "file": Path(name).name,
    }


def build_metadata(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Metadados a partir dos nomes + rótulo afetivo do RAB2 (data/rotulos_afetivos_rab2.csv); idade/MEEM vazios."""
    rows = [parse_filename(p.name) for p in sorted(raw_dir.glob("*.csv"))]
    meta = pd.DataFrame(rows).sort_values("subject_num").reset_index(drop=True)
    meta["diagnosis"] = (meta["group"] == "TCM").astype(int)  # 0=CTR, 1=TCM (como na tese)
    # P40 veio vazio (1 byte) no export: marcamos arquivos sem sinal para ficarem fora das análises.
    meta["file_bytes"] = [(raw_dir / f).stat().st_size for f in meta["file"]]
    meta["usable"] = meta["file_bytes"] > 1_000
    for col in ["age", "meem", "polarity_rab2"]:
        meta[col] = pd.NA
    # Estado afetivo do RAB2 por participante (orientadora, 09/10/2026): regra da tese sobre a nota de polaridade
    # dada à música logo antes do RAB2 (>3 positivo, 3 neutro, <3 negativo).
    if AFFECT_PATH.exists():
        aff = pd.read_csv(AFFECT_PATH).set_index("subject")["affect_rab2"]
        meta["affect_rab2"] = meta["subject"].map(aff)
    else:
        meta["affect_rab2"] = pd.NA
    meta["affect_pos"] = meta["affect_rab2"].map({"positivo": "positivo", "negativo": "nao_positivo",
                                                  "neutro": "nao_positivo"})
    return meta


def load_metadata(only_usable: bool = True) -> pd.DataFrame:
    meta = pd.read_csv(METADATA_PATH) if METADATA_PATH.exists() else build_metadata()
    if only_usable:
        meta = meta[meta["usable"]].reset_index(drop=True)
    return meta


def _parquet_path(subject: str) -> Path:
    return PROCESSED_DIR / "signals" / f"{subject}.parquet"


def load_subject(subject: str, meta: pd.DataFrame | None = None, use_cache: bool = True) -> np.ndarray:
    """Retorna o sinal (n_amostras, 21) em float32 (µV). Usa cache parquet quando existe."""
    cache = _parquet_path(subject)
    if use_cache and cache.exists():
        return pd.read_parquet(cache).to_numpy(np.float32)
    meta = load_metadata() if meta is None else meta
    fname = meta.loc[meta["subject"] == subject, "file"].item()
    df = pd.read_csv(RAW_DIR / fname, header=None, dtype=np.float32)
    if df.shape[1] != N_CHANNELS:
        raise ValueError(f"{fname}: esperado {N_CHANNELS} colunas, veio {df.shape[1]}")
    df.columns = CHANNELS
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
    return df.to_numpy(np.float32)


def iter_subjects(meta: pd.DataFrame | None = None):
    meta = load_metadata() if meta is None else meta
    for row in meta.itertuples(index=False):
        yield row, load_subject(row.subject, meta)
