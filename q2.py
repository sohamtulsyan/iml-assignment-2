import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datasets import load_dataset


#Data loading
def load_iris():
    ds = load_dataset("scikit-learn/iris", split="train")
    feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    X = np.array([[row[f] for f in feature_cols] for row in ds], dtype=np.float64)
    y = np.array([row["Species"] for row in ds])   # ground-truth labels (for reference only)
    return X, y, feature_cols


#kmeans class that will be called as the model
def kmeans(X, k, max_iter=300, tol=1e-6, seed=42):
    rng = np.random.default_rng(seed)

    #Something called kmeans++. Initialization that results in better convergence. 
    centroids = _kmeans_plusplus_init(X, k, rng)

    history = []
    labels  = np.zeros(len(X), dtype=int)

    for _ in range(max_iter):
        #Assigning points to clusters
        dists  = _pairwise_sq_dist(X, centroids)   # (n, k)
        labels = np.argmin(dists, axis=1)

        #Computing inertia - the sum of squared distances intracluster. 
        inertia = sum(
            dists[labels == j, j].sum() for j in range(k)
        )
        history.append(inertia)

        #updation
        new_centroids = np.array([
            X[labels == j].mean(axis=0) if (labels == j).any() else centroids[j]
            for j in range(k)
        ])

        #run till convergence
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    return labels, centroids, inertia, history


def _kmeans_plusplus_init(X, k, rng):
    #Initialization, as found from the internet. 
    n = len(X)
    first = rng.integers(0, n)
    centroids = [X[first]]

    for _ in range(k - 1):
        sq_dists = np.array([
            min(np.sum((x - c) ** 2) for c in centroids)
            for x in X
        ])
        probs = sq_dists / sq_dists.sum()
        next_idx = rng.choice(n, p=probs)
        centroids.append(X[next_idx])

    return np.array(centroids)


def _pairwise_sq_dist(X, C):
    #Just euclidean distance between points. 
    return np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)


def run_all(X, K_values=(2, 3, 4, 5), n_restarts=10):
    #Running kmeans for various values of k. 
    results = {}
    for k in K_values:
        best = None
        for seed in range(n_restarts):
            labels, centroids, inertia, history = kmeans(X, k, seed=seed)
            if best is None or inertia < best["inertia"]:
                best = {
                    "labels":    labels,
                    "centroids": centroids,
                    "inertia":   inertia,
                    "history":   history,
                }
        print(f"  k={k}  inertia={best['inertia']:.4f}  "
              f"iters={len(best['history'])}")
        results[k] = best
    return results


#Visualisations

PLOT_PAIRS = [(0, 2), (0, 1)] 
FEATURE_NAMES = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]

PALETTE = [
    "#e63946", "#2196f3", "#4caf50", "#ff9800", "#9c27b0"
]


def plot_clusters(results, X, K_values, feature_names):
    n_pairs = len(PLOT_PAIRS)
    n_k     = len(K_values)

    fig, axes = plt.subplots(n_k, n_pairs,
                             figsize=(6 * n_pairs, 4.5 * n_k))

    for row, k in enumerate(K_values):
        labels    = results[k]["labels"]
        centroids = results[k]["centroids"]
        inertia   = results[k]["inertia"]

        for col, (fx, fy) in enumerate(PLOT_PAIRS):
            ax = axes[row, col]
            colors = [PALETTE[l] for l in labels]

            ax.scatter(X[:, fx], X[:, fy],
                       c=colors, alpha=0.7, s=40, edgecolors="white", linewidths=0.4)

            # Centroids
            ax.scatter(centroids[:, fx], centroids[:, fy],
                       c="black", marker="X", s=180, zorder=5,
                       edgecolors="white", linewidths=1, label="Centroid")

            ax.set_xlabel(feature_names[fx], fontsize=10)
            ax.set_ylabel(feature_names[fy], fontsize=10)
            ax.set_title(f"k={k}  |  Inertia={inertia:.2f}", fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.25, linestyle="--")

            # Legend patches
            patches = [mpatches.Patch(color=PALETTE[j], label=f"Cluster {j+1}")
                       for j in range(k)]
            patches.append(mpatches.Patch(color="black", label="Centroid"))
            ax.legend(handles=patches, fontsize=8, loc="upper right")

    plt.suptitle("K-Means Clusters on Iris Dataset",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("kmeans_clusters.png", dpi=150, bbox_inches="tight")
    print("Saved: kmeans_clusters.png")
    plt.show()


def plot_inertia_curve(results, K_values):
    #To find optimal number of K
    inertias = [results[k]["inertia"] for k in K_values]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(K_values, inertias, marker="o", color="#2196f3",
            linewidth=2.2, markersize=9, markerfacecolor="white",
            markeredgewidth=2.5)

    for k, val in zip(K_values, inertias):
        ax.annotate(f"{val:.1f}",
                    xy=(k, val), xytext=(0, 12),
                    textcoords="offset points",
                    ha="center", fontsize=9, color="#333")

    ax.set_xticks(K_values)
    ax.set_xlabel("Number of Clusters  k", fontsize=12)
    ax.set_ylabel("Total Discrepancy (Inertia)", fontsize=12)
    ax.set_title("Elbow Curve – K-Means on Iris Dataset",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("kmeans_inertia.png", dpi=150, bbox_inches="tight")
    print("Saved: kmeans_inertia.png")
    plt.show()


def plot_convergence(results, K_values):
    """Inertia vs iteration for each k."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k in K_values:
        hist = results[k]["history"]
        ax.plot(range(1, len(hist) + 1), hist,
                marker=".", label=f"k={k}", linewidth=1.8)

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Inertia", fontsize=11)
    ax.set_title("Convergence of K-Means per k", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("kmeans_convergence.png", dpi=150, bbox_inches="tight")
    print("Saved: kmeans_convergence.png")
    plt.show()


def print_summary(results, K_values):
    print("\n" + "═" * 52)
    print(f"{'k':>4}  {'Inertia':>12}  {'Iterations':>12}")
    print("─" * 52)
    for k in K_values:
        r = results[k]
        print(f"{k:>4}  {r['inertia']:>12.4f}  {len(r['history']):>12}")
    print("═" * 52)

#main to call all the models
if __name__ == "__main__":
    K_VALUES = [2, 3, 4, 5]

    print("Loading Iris dataset …")
    X, y, feature_cols = load_iris()
    print(f"  Shape: {X.shape}")

    print("\nRunning K-Means (10 restarts per k) …")
    results = run_all(X, K_VALUES, n_restarts=10)

    print_summary(results, K_VALUES)
    plot_inertia_curve(results, K_VALUES)
    plot_convergence(results, K_VALUES)
    plot_clusters(results, X, K_VALUES, FEATURE_NAMES)
