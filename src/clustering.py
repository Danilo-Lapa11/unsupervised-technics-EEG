"""Protocolo de clusterização dos baselines (extraído do notebook 04 para reuso nos notebooks 04 e 07).

Clusterização sem rótulos; associação com grupo (CTR/TCM) e sexo só a posteriori.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .evaluation import internal_metrics, external_metrics, majority_vote, bootstrap_stability

SEED = 0


def to_pca(X, var: float = 0.95, seed: int = SEED) -> np.ndarray:
    """StandardScaler -> PCA retendo `var` da variância."""
    Xs = StandardScaler().fit_transform(X)
    return PCA(n_components=var, random_state=seed).fit_transform(Xs)


def balance_windows(feat_w: pd.DataFrame, n_per_subject: int = 40, seed: int = SEED) -> pd.DataFrame:
    """Até `n_per_subject` janelas por participante (neutraliza a duração, confundida com o grupo)."""
    return (feat_w.sample(frac=1, random_state=seed).groupby("subject").head(n_per_subject)
            .sort_values(["subject", "window"]).reset_index(drop=True))


def dbscan_eps(X, k: int = 5) -> float:
    d = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)[0][:, -1]
    return float(np.quantile(d, 0.9))


def algorithms(k: int, seed: int = SEED) -> dict:
    return {
        "random": lambda Z: np.random.default_rng(seed).integers(0, k, len(Z)),
        "kmeans": lambda Z: KMeans(k, n_init=20, random_state=seed).fit_predict(Z),
        "agglo_ward": lambda Z: AgglomerativeClustering(k, linkage="ward").fit_predict(Z),
        "gmm": lambda Z: GaussianMixture(k, covariance_type="diag", n_init=5, random_state=seed).fit_predict(Z),
        "spectral": lambda Z: SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=min(10, len(Z) - 1),
                                                 random_state=seed).fit_predict(Z),
    }


def density_algorithms(X) -> dict:
    eps = dbscan_eps(X)
    mcs = max(5, len(X) // 50)
    return {
        "dbscan": lambda Z: DBSCAN(eps=eps, min_samples=5).fit_predict(Z),
        "hdbscan": lambda Z: HDBSCAN(min_cluster_size=mcs).fit_predict(Z),
    }


def k_sweep(X, ks=range(2, 9), seed: int = SEED) -> pd.DataFrame:
    """Silhouette/Davies-Bouldin do k-means e BIC do GMM para k = 2..8 (escolha de k sem rótulos)."""
    rows = []
    for k in ks:
        lab = KMeans(k, n_init=20, random_state=seed).fit_predict(X)
        gmm = GaussianMixture(k, covariance_type="diag", n_init=3, random_state=seed).fit(X)
        rows.append({"k": k, **internal_metrics(X, lab), "gmm_bic": gmm.bic(X)})
    return pd.DataFrame(rows)


def best_k(sweep: pd.DataFrame) -> int:
    return int(sweep.loc[sweep.silhouette.idxmax(), "k"])


def evaluate(level: str, X, meta: pd.DataFrame, k: int, subjects=None, density: bool | None = None,
             stability: bool = True) -> tuple[list[dict], dict]:
    """Roda os algoritmos e mede qualidade interna + associação a posteriori.

    meta: tabela alinhada com as linhas de X (colunas group, sex[, subject]).
    subjects: se dado (nível janela), mede também o voto majoritário por participante e o ARI com a identidade.
    """
    rows, labels_out = [], {}
    density = (k == 2) if density is None else density
    algos = {**algorithms(k), **(density_algorithms(X) if density else {})}
    for name, fp in algos.items():
        lab = fp(X)
        row = {"level": level, "algo": name, "k_param": k if name not in ("dbscan", "hdbscan") else "auto",
               **internal_metrics(X, lab)}
        if subjects is None:
            row |= external_metrics(meta.group, lab, "grupo_") | external_metrics(meta.sex, lab, "sexo_")
        else:
            row |= external_metrics(meta.group, lab, "grupo_")
            vote = majority_vote(subjects, lab)
            sm = meta.drop_duplicates("subject").set_index("subject").loc[vote.index]
            row |= external_metrics(sm.group, vote.values, "grupo_voto_")
            row |= {"subject_ARI": external_metrics(subjects, lab)["ARI"]}  # quanto os clusters "acham pessoas"
        if stability and name in ("kmeans", "agglo_ward", "gmm"):
            row["stability_ARI"] = bootstrap_stability(fp, X, n_boot=10)
        rows.append(row)
        labels_out[name] = lab
    return rows, labels_out
