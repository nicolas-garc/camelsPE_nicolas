"""Embed and cluster the per-(pair, parameter) feature vectors.

Runs as the last stage of run_sweep.py, or standalone once the sweep has
written sweep_output/features/:

    python cluster_params.py [--out sweep_output] [--k-range 2 12]

The features (src/features.py) are the numeric content of each consolidated
summary figure, so clustering them sorts (pair, parameter) combinations into
groups that share a constraint structure — the algorithmic version of the
six-case taxonomy in CLAUDE.md. Every point keeps its pair/parameter identity,
so each cluster maps back to the figures that produced it: medoid figures are
copied into clusters/medoids/ for exactly that check.

Outputs (in <out>/clusters/):
    features_all.csv     every feature row + rule-based case label
    clusters.csv         cluster id, PCA coords, 2D embedding per row
    k_selection.png      silhouette vs k — how many groups the data supports
    embedding_*.png      2D embedding colored by cluster / rule label / R²
    cluster_profile.png  mean z-scored feature per cluster (what defines each)
    cluster_vs_rule.png  cross-tab against the hand taxonomy
    medoids/             the summary figure of each cluster's most typical rows

UMAP and HDBSCAN are used when installed; otherwise it falls back to
scikit-learn's t-SNE and KMeans, so the stage never blocks an HPC run.
"""
import os
import sys
import glob
import shutil
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import features as features_mod

ID_COLS = ["pair", "obs1", "obs2", "param"]
N_MEDOIDS = 3      # exemplar figures copied per cluster


def load_features(out_dir):
    paths = sorted(glob.glob(os.path.join(out_dir, "features", "*.csv")))
    if not paths:
        raise SystemExit(f"no feature CSVs in {os.path.join(out_dir, 'features')} — run the sweep first")
    df = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
    print(f"[cluster] {len(df)} rows from {len(paths)} pairs")
    return df


def feature_matrix(df):
    """Numeric columns only, inf -> nan -> column median, then z-scored."""
    X = df.drop(columns=[c for c in ID_COLS if c in df], errors="ignore")
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    keep = X.columns[X.std(axis=0) > 1e-12]          # constant columns carry no signal
    X = X[keep]
    return X, StandardScaler().fit_transform(X.values)


def choose_k(Z, k_range):
    """KMeans over a range of k; pick the best average silhouette."""
    rows = []
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z)
        rows.append({"k": k, "silhouette": silhouette_score(Z, km.labels_),
                     "inertia": km.inertia_})
    scores = pd.DataFrame(rows)
    best = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    print(f"[cluster] silhouette picks k={best}")
    return best, scores


def embed_2d(Z, seed=0):
    """UMAP when available (better global structure), else t-SNE."""
    try:
        import umap
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(Z), "UMAP"
    except Exception:
        perp = min(30, max(5, (len(Z) - 1) // 3))
        return TSNE(n_components=2, random_state=seed, init="pca",
                    perplexity=perp).fit_transform(Z), "t-SNE"


def cluster_labels(Z, k):
    """HDBSCAN when available (finds its own k, marks outliers), else KMeans."""
    try:
        import hdbscan
        lab = hdbscan.HDBSCAN(min_cluster_size=max(10, len(Z) // 100)).fit_predict(Z)
        if len(set(lab) - {-1}) > 1:
            print(f"[cluster] HDBSCAN: {len(set(lab) - {-1})} clusters, "
                  f"{(lab == -1).sum()} unassigned")
            return lab, "HDBSCAN"
    except Exception:
        pass
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z), "KMeans"


def scatter(emb, color, title, path, discrete=True, cmap="tab10"):
    fig, ax = plt.subplots(figsize=(8, 7))
    if discrete:
        for v in pd.unique(color):
            m = color == v
            ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.75, label=str(v))
        ax.legend(fontsize=7, markerscale=1.6, loc="best", ncol=2)
    else:
        s = ax.scatter(emb[:, 0], emb[:, 1], c=color, s=8, alpha=0.8, cmap=cmap)
        fig.colorbar(s, ax=ax)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def copy_medoids(df, X, labels, out_dir, cdir):
    """Copy each cluster's most typical summary figures, so a cluster can be
    named by looking at the plots it actually contains."""
    mdir = os.path.join(cdir, "medoids")
    os.makedirs(mdir, exist_ok=True)
    rows = []
    for c in sorted(set(labels) - {-1}):
        m = labels == c
        sub = X[m]
        center = sub.mean(axis=0)
        order = np.argsort(((sub - center) ** 2).sum(axis=1))[:N_MEDOIDS]
        for rank, i in enumerate(order, 1):
            r = df[m].iloc[i]
            src = os.path.join(out_dir, "plots", r["pair"], f"{r['param']}_summary.png")
            dst = os.path.join(mdir, f"cluster{c:02d}_{rank}_{r['pair']}_{r['param']}.png")
            if os.path.exists(src):
                shutil.copyfile(src, dst)
            rows.append({"cluster": c, "rank": rank, "pair": r["pair"], "param": r["param"],
                         "figure": src, "copied": os.path.exists(src)})
    pd.DataFrame(rows).to_csv(os.path.join(cdir, "medoids.csv"), index=False)
    n = sum(r["copied"] for r in rows)
    print(f"[cluster] {n}/{len(rows)} medoid figures -> {mdir}")


def run(out_dir, k_range=(2, 12)):
    cdir = os.path.join(out_dir, "clusters")
    os.makedirs(cdir, exist_ok=True)

    df = load_features(out_dir)
    df["rule_case"] = features_mod.rule_based_case(df)
    df.to_csv(os.path.join(cdir, "features_all.csv"), index=False)

    X, Z = feature_matrix(df)
    print(f"[cluster] {Z.shape[1]} features after dropping constant columns")

    # PCA first: denoises and decorrelates before clustering/embedding
    n_pc = int(min(10, Z.shape[1], Z.shape[0] - 1))
    pca = PCA(n_components=n_pc, random_state=0)
    P = pca.fit_transform(Z)
    evr = pca.explained_variance_ratio_
    print(f"[cluster] PCA {n_pc} comps explain {evr.sum():.1%} "
          f"(PC1 {evr[0]:.1%}, PC2 {evr[1]:.1%})")

    k, scores = choose_k(P, k_range)
    scores.to_csv(os.path.join(cdir, "k_selection.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(scores["k"], scores["silhouette"], "o-")
    ax[0].axvline(k, color="r", ls="--", lw=1, label=f"chosen k={k}")
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("mean silhouette")
    ax[0].legend()
    ax[1].plot(scores["k"], scores["inertia"], "o-")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("inertia")
    fig.suptitle("How many groups does the feature space support?")
    fig.tight_layout()
    fig.savefig(os.path.join(cdir, "k_selection.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    labels, algo = cluster_labels(P, k)
    emb, emb_name = embed_2d(P)
    df_out = df[ID_COLS].copy()
    df_out["cluster"] = labels
    df_out["rule_case"] = df["rule_case"]
    df_out[["emb_x", "emb_y"]] = emb
    for i in range(min(4, n_pc)):
        df_out[f"pc{i+1}"] = P[:, i]
    df_out.to_csv(os.path.join(cdir, "clusters.csv"), index=False)

    scatter(emb, df_out["cluster"].values, f"{emb_name} embedding — {algo} clusters",
            os.path.join(cdir, "embedding_clusters.png"))
    scatter(emb, df["rule_case"].values, f"{emb_name} embedding — rule-based case",
            os.path.join(cdir, "embedding_rule_case.png"))
    for col, title in [("r2_both_clean", "aligned R² (both clean)"),
                       ("gain_from_combining", "gain from combining"),
                       ("allegiance", "allegiance (0 = obs2, 1 = obs1)")]:
        if col in df:
            scatter(emb, df[col].values, f"{emb_name} embedding — {title}",
                    os.path.join(cdir, f"embedding_{col}.png"), discrete=False, cmap="viridis")

    # What defines each cluster: mean z-scored feature value
    prof = pd.DataFrame(Z, columns=X.columns).groupby(labels).mean()
    prof.index.name = "cluster"
    prof.to_csv(os.path.join(cdir, "cluster_profile.csv"))
    fig = plt.figure(figsize=(max(12, 0.32 * prof.shape[1]), 0.5 * prof.shape[0] + 3))
    sns.heatmap(prof, cmap="RdBu_r", center=0, linewidths=0.2,
                cbar_kws={"label": "mean z-scored feature"})
    plt.title(f"Cluster profiles ({algo}) — what makes each group different")
    plt.tight_layout()
    fig.savefig(os.path.join(cdir, "cluster_profile.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    ct = pd.crosstab(df_out["cluster"], df["rule_case"])
    ct.to_csv(os.path.join(cdir, "cluster_vs_rule.csv"))
    fig = plt.figure(figsize=(max(7, 0.9 * ct.shape[1] + 3), 0.5 * ct.shape[0] + 3))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", linewidths=0.3)
    plt.title("Clusters vs the hand taxonomy")
    plt.tight_layout()
    fig.savefig(os.path.join(cdir, "cluster_vs_rule.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    copy_medoids(df, Z, np.asarray(labels), out_dir, cdir)
    sizes = pd.Series(labels).value_counts().sort_index()
    print(f"[cluster] {algo} cluster sizes: {sizes.to_dict()}")
    print(f"[cluster] -> {cdir}")
    return df_out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "sweep_output"))
    ap.add_argument("--k-range", type=int, nargs=2, default=[2, 12])
    args = ap.parse_args()
    run(args.out, tuple(args.k_range))


if __name__ == "__main__":
    main()
