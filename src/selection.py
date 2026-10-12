"""Seleção de atributos.

- Não supervisionada (resultado principal; nenhum rótulo é usado): quase-constantes -> Laplacian Score -> poda de
  redundância -> tamanho escolhido pelo silhouette do k-means.
- Supervisionada aninhada (referência/teto): mRMR dentro de cada dobra leave-one-subject-out. Os rótulos só são
  vistos no treino de cada dobra, então a AUC não tem vazamento; já clusterizar com os atributos escolhidos dessa forma
  é otimista (a escolha foi guiada pelo diagnóstico) e deve ser apresentado como tal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0


# ------------------------------------------------------------------------------------------- não supervisionada

def drop_quasi_constant(df: pd.DataFrame, min_unique: int = 5, rel_iqr: float = 1e-6) -> list[str]:
    """Colunas com variação útil (descarta constantes, quase-binárias e IQR ~ 0)."""
    keep = []
    for c in df.columns:
        x = df[c].to_numpy(float)
        x = x[np.isfinite(x)]
        if len(np.unique(x)) < min_unique:
            continue
        q75, q25 = np.percentile(x, [75, 25])
        if (q75 - q25) <= rel_iqr * (np.abs(np.median(x)) + 1e-12):
            continue
        keep.append(c)
    return keep


def laplacian_score(X: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    """Laplacian Score (He, Cai & Niyogi, 2005): quanto cada atributo preserva a vizinhança local dos dados.

    Menor = melhor. Grafo kNN simétrico com kernel gaussiano (t = média das distâncias² entre vizinhos)."""
    Z = StandardScaler().fit_transform(X)
    D2 = cdist(Z, Z, "sqeuclidean")
    n = len(Z)
    nn = np.argsort(D2, axis=1)[:, 1:n_neighbors + 1]
    mask = np.zeros_like(D2, bool)
    mask[np.repeat(np.arange(n), n_neighbors), nn.ravel()] = True
    mask |= mask.T
    t = D2[mask].mean()
    W = np.where(mask, np.exp(-D2 / t), 0.0)
    d = W.sum(axis=1)
    L = np.diag(d) - W
    scores = np.empty(Z.shape[1])
    for j in range(Z.shape[1]):
        f = Z[:, j]
        f = f - (f @ d) / d.sum()
        den = f @ (d * f)
        scores[j] = (f @ L @ f) / den if den > 1e-12 else np.inf
    return scores


def spearman_abs(df: pd.DataFrame) -> pd.DataFrame:
    R = np.apply_along_axis(rankdata, 0, df.to_numpy(float))
    return pd.DataFrame(np.abs(np.corrcoef(R.T)), index=df.columns, columns=df.columns)


def prune_redundant(df: pd.DataFrame, order: list[str], threshold: float = 0.9) -> list[str]:
    """Percorre `order` (melhores primeiro) e mantém um atributo se |Spearman| < threshold com todos os já mantidos."""
    C = spearman_abs(df[order])
    kept = []
    for c in order:
        if all(C.at[c, k] < threshold for k in kept):
            kept.append(c)
    return kept


def select_unsupervised(df: pd.DataFrame, sizes=(10, 20, 40, 80, 160), cluster_ks=range(2, 7),
                        redundancy: float = 0.9) -> tuple[list[str], pd.DataFrame]:
    """Retorna (atributos escolhidos, tabela da varredura). df: participantes x atributos (sem rótulos).

    Atenção: o silhouette tende a favorecer subconjuntos pequenos (menos dimensões) — a tabela mostra todos os
    tamanhos para inspecionar esse viés."""
    cols = drop_quasi_constant(df)
    ls = pd.Series(laplacian_score(df[cols].to_numpy()), index=cols).sort_values()
    ranked = prune_redundant(df, list(ls.index), redundancy)
    rows = []
    for m in sizes:
        sub = ranked[:m]
        if len(sub) < m:
            break
        Z = StandardScaler().fit_transform(df[sub])
        if m > 10:
            Z = PCA(0.95, random_state=SEED).fit_transform(Z)
        for k in cluster_ks:
            lab = KMeans(k, n_init=20, random_state=SEED).fit_predict(Z)
            rows.append({"n_atributos": m, "k": k, "silhouette": silhouette_score(Z, lab)})
    sweep = pd.DataFrame(rows)
    best = sweep.loc[sweep.silhouette.idxmax()]
    sweep.attrs.update(n_quase_constantes=len(df.columns) - len(cols), n_apos_poda=len(ranked),
                       laplacian=ls)
    return ranked[: int(best.n_atributos)], sweep


# ------------------------------------------------------------------------------------- supervisionada aninhada

def mrmr(X: pd.DataFrame, y: np.ndarray, m: int = 20, prefilter: int = 200, seed: int = SEED) -> list[str]:
    """mRMR (diferença): relevância = informação mútua com y; redundância = |Spearman| médio com os já escolhidos."""
    rel = pd.Series(mutual_info_classif(X, y, random_state=seed), index=X.columns)
    cand = list(rel.nlargest(min(prefilter, len(rel))).index)
    C = spearman_abs(X[cand])
    chosen = [cand[0]]
    while len(chosen) < m:
        rest = [c for c in cand if c not in chosen]
        score = rel[rest] - C.loc[rest, chosen].mean(axis=1)
        chosen.append(score.idxmax())
    return chosen


def nested_mrmr_loso(X: pd.DataFrame, y: np.ndarray, m: int = 20) -> tuple[np.ndarray, pd.Series]:
    """Leave-one-subject-out com mRMR refeito em cada dobra (sem vazamento). X: participantes x atributos.

    Retorna (probabilidade de TCM fora da dobra, frequência de seleção de cada atributo entre as dobras)."""
    proba = np.empty(len(y))
    counts = pd.Series(0, index=X.columns)
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        sel = mrmr(X[tr], y[tr], m)
        counts[sel] += 1
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
        model.fit(X.loc[tr, sel], y[tr])
        proba[i] = model.predict_proba(X.loc[~tr, sel])[0, 1]
    return proba, (counts / len(y)).sort_values(ascending=False)
