"""Métricas internas de clusterização e associação a posteriori com rótulos."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics


def internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    mask = labels >= 0  # ignora ruído (-1) de DBSCAN/HDBSCAN
    n_clusters = len(np.unique(labels[mask]))
    out = {"n_clusters": n_clusters, "noise_frac": float((~mask).mean())}
    if n_clusters < 2 or mask.sum() <= n_clusters:
        return out | {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    Xm, lm = X[mask], labels[mask]
    sample = min(len(Xm), 10000)
    return out | {
        "silhouette": metrics.silhouette_score(Xm, lm, sample_size=sample, random_state=0),
        "davies_bouldin": metrics.davies_bouldin_score(Xm, lm),
        "calinski_harabasz": metrics.calinski_harabasz_score(Xm, lm),
    }


def purity(y_true, labels) -> float:
    ct = pd.crosstab(labels, y_true)
    return ct.max(axis=1).sum() / ct.values.sum()


def external_metrics(y_true, labels, prefix: str = "") -> dict:
    y_true, labels = np.asarray(y_true), np.asarray(labels)
    ct = pd.crosstab(labels, y_true)
    if ct.shape[0] > 1 and ct.shape[1] > 1:
        if ct.shape == (2, 2):
            p = stats.fisher_exact(ct.values)[1]
        else:
            p = stats.chi2_contingency(ct.values)[1]
    else:
        p = np.nan
    return {
        f"{prefix}ARI": metrics.adjusted_rand_score(y_true, labels),
        f"{prefix}NMI": metrics.normalized_mutual_info_score(y_true, labels),
        f"{prefix}V": metrics.v_measure_score(y_true, labels),
        f"{prefix}purity": purity(y_true, labels),
        f"{prefix}p_assoc": p,
    }


def majority_vote(subjects, labels) -> pd.Series:
    """Rótulo de cluster por participante a partir das janelas (ignora ruído)."""
    df = pd.DataFrame({"subject": subjects, "label": labels})
    df = df[df["label"] >= 0]
    return df.groupby("subject")["label"].agg(lambda s: s.value_counts().idxmax())


def bootstrap_stability(fit_predict, X: np.ndarray, n_boot: int = 20, frac: float = 0.8, seed: int = 0) -> float:
    """ARI médio entre o particionamento completo e reajustes em subamostras (nos pontos em comum)."""
    rng = np.random.default_rng(seed)
    ref = fit_predict(X)
    aris = []
    for _ in range(n_boot):
        idx = rng.choice(len(X), int(frac * len(X)), replace=False)
        aris.append(metrics.adjusted_rand_score(ref[idx], fit_predict(X[idx])))
    return float(np.mean(aris))


def mann_whitney_fdr(df: pd.DataFrame, group: pd.Series, a="CTR", b="TCM") -> pd.DataFrame:
    """Teste Mann-Whitney por coluna, efeito (rank-biserial) e correção FDR (Benjamini-Hochberg)."""
    rows = []
    for col in df.columns:
        xa, xb = df.loc[group == a, col].dropna(), df.loc[group == b, col].dropna()
        u, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        rows.append({"feature": col, "median_" + a: xa.median(), "median_" + b: xb.median(),
                     "rank_biserial": 1 - 2 * u / (len(xa) * len(xb)), "p": p})
    res = pd.DataFrame(rows)
    res["q_fdr"] = stats.false_discovery_control(res["p"].to_numpy())
    return res.sort_values("p").reset_index(drop=True)


# ------------------------------------------------------------------ comparação de variantes de pré-processamento

SCALE_DEPENDENT = ("abs_", "hjorth_activity", "std")


def scale_invariant(feats) -> list[str]:
    """Features que não mudam se o sinal inteiro for multiplicado por uma constante (ganho/impedância)."""
    return [c for c in feats if not c.split("__")[0].startswith(SCALE_DEPENDENT)]


def icc1(df: pd.DataFrame, groups: pd.Series) -> pd.Series:
    """ICC(1) por coluna (ANOVA de um fator, grupos desbalanceados): fração da variância entre participantes.

    Substitui a razão simples var(médias)/var(total) usada no notebook 03, que podia passar de 1."""
    out = {}
    g = groups.to_numpy()
    labels, n_i = np.unique(g, return_counts=True)
    N, k = len(g), len(labels)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    for col in df.columns:
        x = df[col].to_numpy(float)
        grand = np.nanmean(x)
        means = pd.Series(x).groupby(g).mean().reindex(labels).to_numpy()
        msb = (n_i * (means - grand) ** 2).sum() / (k - 1)
        msw = ((x - means[np.searchsorted(labels, g)]) ** 2).sum() / (N - k)
        out[col] = (msb - msw) / (msb + (n0 - 1) * msw)
    return pd.Series(out)


def auc_ci(y, p, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """IC95% da AUC por bootstrap de participantes."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    a = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            a.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def loso_logreg(X, y) -> np.ndarray:
    """Probabilidades leave-one-subject-out (nível participante) de uma regressão logística regularizada com PCA."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), PCA(0.95, random_state=0),
                          LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
    return cross_val_predict(model, X, y, cv=LeaveOneOut(), method="predict_proba")[:, 1]
