import urllib.request
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize


def load_breast_cancer():
    #loading the dataset. Hope this is the right one. 
    url = (
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    )
    print("Downloading Breast Cancer Wisconsin dataset from UCI …")
    with urllib.request.urlopen(url) as resp:
        raw = resp.read().decode("utf-8")

    rows = []
    for line in raw.strip().splitlines():
        parts = line.split(",")
        label = 1.0 if parts[1].strip() == "B" else -1.0 
        features = [float(v) for v in parts[2:]]
        rows.append([label] + features)

    data = np.array(rows, dtype=np.float64)
    y = data[:, 0]
    X = data[:, 1:]

    print(f"  Loaded: X={X.shape}, y={y.shape}")
    print(f"  Benign (+1): {(y==1).sum()}   Malignant (-1): {(y==-1).sum()}")
    return X, y


#Just some helper functions to help with preprocessing
def standardise(X_train, X_test):
    mu  = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mu) / std, (X_test - mu) / std, mu, std


def train_test_split(X, y, test_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


#Doing PCA but just for visualisation - flattening to 2 dims
def pca_2d(X_train, X_test):
    mu = X_train.mean(axis=0)
    Xc = X_train - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V2 = Vt[:2].T                          # (d, 2)
    return (X_train - mu) @ V2, (X_test - mu) @ V2


#business logic
class SoftMarginSVM:
    def __init__(self, C=1.0):
        self.C = C
        self.alpha      = None   # dual variables
        self.support_vectors   = None
        self.support_labels    = None
        self.support_alphas    = None
        self.w          = None
        self.b          = None

    @staticmethod
    def _build_Q(X, y):
        K = X @ X.T          # (n, n) linear kernel
        return np.outer(y, y) * K

    def fit(self, X, y, sv_tol=1e-5):
        n = len(y)
        C = self.C
        Q = self._build_Q(X, y)

        # Objective: minimise f(α) = ½ αᵀ Q α − 1ᵀα
        def objective(alpha):
            return 0.5 * alpha @ Q @ alpha - np.sum(alpha)

        def gradient(alpha):
            return Q @ alpha - np.ones(n)

        # Equality constraint: Σ αᵢ yᵢ = 0
        constraints = {"type": "eq",
                       "fun":  lambda a: np.dot(a, y),
                       "jac":  lambda a: y}

        # Box constraints: 0 ≤ αᵢ ≤ C
        bounds = [(0.0, C)] * n

        # Initial point: all zeros
        alpha0 = np.zeros(n)

        result = minimize(
            objective,
            alpha0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        if not result.success:
            print(f"  [Warning] Solver message: {result.message}")

        self.alpha = np.maximum(result.x, 0.0)   # clip numerical noise

        #identifying support vectors
        sv_mask = self.alpha > sv_tol
        self.support_vectors = X[sv_mask]
        self.support_labels  = y[sv_mask]
        self.support_alphas  = self.alpha[sv_mask]

        self.w = (self.support_alphas * self.support_labels) @ self.support_vectors

        margin_mask = (self.support_alphas > sv_tol) & (self.support_alphas < C - sv_tol)
        if margin_mask.any():
            self.b = np.mean(
                self.support_labels[margin_mask]
                - self.support_vectors[margin_mask] @ self.w
            )
        else:
            # Fallback: average over all support vectors
            self.b = np.mean(
                self.support_labels - self.support_vectors @ self.w
            )

        return self

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

    @property
    def n_support_vectors(self):
        return len(self.support_vectors) if self.support_vectors is not None else 0



#Trying out different values of C for experiment's sakes.
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def run_experiments(X_tr, X_te, y_tr, y_te):
    results = {}
    for C in C_VALUES:
        print(f"  Training SVM with C={C} …")
        svm = SoftMarginSVM(C=C)
        svm.fit(X_tr, y_tr)
        tr_acc = svm.accuracy(X_tr, y_tr)
        te_acc = svm.accuracy(X_te, y_te)
        nsv    = svm.n_support_vectors
        print(f"    Train acc={tr_acc:.4f}  Test acc={te_acc:.4f}  "
              f"#SV={nsv}")
        results[C] = {
            "svm":      svm,
            "train_acc": tr_acc,
            "test_acc":  te_acc,
            "n_sv":      nsv,
        }
    return results


#visualisations
def plot_metrics_vs_C(results):
    C_vals   = list(results.keys())
    tr_accs  = [results[c]["train_acc"] for c in C_vals]
    te_accs  = [results[c]["test_acc"]  for c in C_vals]
    n_svs    = [results[c]["n_sv"]      for c in C_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    #Testing
    x = np.arange(len(C_vals))
    ax1.plot(x, tr_accs, marker="o", label="Train Accuracy",
             color="#2196f3", linewidth=2)
    ax1.plot(x, te_accs, marker="s", label="Test Accuracy",
             color="#e63946", linewidth=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(c) for c in C_vals])
    ax1.set_xlabel("Margin Cost  C", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("Accuracy vs C", fontsize=12, fontweight="bold")
    ax1.set_ylim(0.5, 1.02)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle="--")

    #The support vecs
    bars = ax2.bar(x, n_svs, color="#4caf50", edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, n_svs):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 2, str(v),
                 ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(c) for c in C_vals])
    ax2.set_xlabel("Margin Cost  C", fontsize=11)
    ax2.set_ylabel("Number of Support Vectors", fontsize=11)
    ax2.set_title("Support Vectors vs C", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    plt.suptitle("Soft-Margin SVM on Breast Cancer Dataset",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("svm_metrics.png", dpi=150, bbox_inches="tight")
    print("Saved: svm_metrics.png")
    plt.show()


def plot_decision_boundary_2d(svm_2d, X2_tr, X2_te, y_tr, y_te, C):
    #Uses the PCA logic.
    fig, ax = plt.subplots(figsize=(8, 6))

    # Grid
    x_min, x_max = X2_tr[:, 0].min() - 0.5, X2_tr[:, 0].max() + 0.5
    y_min, y_max = X2_tr[:, 1].min() - 0.5, X2_tr[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_2d.decision_function(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9],
                colors=["#ffcdd2", "#c8e6c9"], alpha=0.5)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1],
               colors=["#e63946", "black", "#2196f3"],
               linewidths=[1.2, 2.0, 1.2],
               linestyles=["--", "-", "--"])

    # Data points (train)
    for lbl, color, marker, name in [
            (1,  "#2196f3", "o", "Benign"),
            (-1, "#e63946", "^", "Malignant")]:
        mask = y_tr == lbl
        ax.scatter(X2_tr[mask, 0], X2_tr[mask, 1],
                   c=color, marker=marker, s=30,
                   alpha=0.6, edgecolors="white", linewidths=0.3,
                   label=f"Train – {name}")

    # Test points
    for lbl, color, marker in [(1, "#2196f3", "o"), (-1, "#e63946", "^")]:
        mask = y_te == lbl
        ax.scatter(X2_te[mask, 0], X2_te[mask, 1],
                   c=color, marker=marker, s=60,
                   edgecolors="black", linewidths=0.8, alpha=0.9)

    # Support vectors
    sv_proj = svm_2d.support_vectors
    ax.scatter(sv_proj[:, 0], sv_proj[:, 1],
               s=180, facecolors="none", edgecolors="gold",
               linewidths=1.8, zorder=5, label="Support Vectors")

    ax.set_xlabel("PCA Component 1", fontsize=11)
    ax.set_ylabel("PCA Component 2", fontsize=11)
    ax.set_title(f"Decision Boundary – Soft-Margin SVM (C={C})\n"
                 f"(PCA 2-D projection)  #SV={svm_2d.n_support_vectors}  "
                 f"Test Acc={svm_2d.accuracy(X2_te, y_te):.3f}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, linestyle="--")

    plt.tight_layout()
    plt.savefig("svm_boundary.png", dpi=150, bbox_inches="tight")
    print("Saved: svm_boundary.png")
    plt.show()


def print_summary(results):
    print("\n" + "═" * 60)
    print(f"{'C':>8}  {'Train Acc':>10}  {'Test Acc':>10}  {'#SV':>8}")
    print("─" * 60)
    for C, r in results.items():
        print(f"{C:>8}  {r['train_acc']:>10.4f}  {r['test_acc']:>10.4f}  {r['n_sv']:>8}")
    print("═" * 60)

    best_C = max(results, key=lambda c: results[c]["test_acc"])
    print(f"\nBest test accuracy: {results[best_C]['test_acc']:.4f}  at C={best_C}")
    print(f"  → #Support vectors: {results[best_C]['n_sv']}")
    print(f"  → ||w||₂ = {np.linalg.norm(results[best_C]['svm'].w):.4f}")
    return best_C



#main function to call everything
if __name__ == "__main__":
    np.random.seed(42)

    # Load & split
    X, y = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_ratio=0.2, seed=42)

    # Standardise (critical for SVM)
    X_tr_s, X_te_s, _, _ = standardise(X_tr, X_te)

    # Run experiments for multiple C values (full 30-D space)
    print(f"\nRunning Soft-Margin SVM for C in {C_VALUES} …")
    results = run_experiments(X_tr_s, X_te_s, y_tr, y_te)

    best_C = print_summary(results)

    # Plot accuracy & #SV vs C
    plot_metrics_vs_C(results)

    # Decision boundary in PCA-2D space for best C
    print(f"\nFitting SVM in PCA-2D space for boundary visualisation (C={best_C}) …")
    X2_tr, X2_te = pca_2d(X_tr_s, X_te_s)
    svm_2d = SoftMarginSVM(C=best_C).fit(X2_tr, y_tr)
    plot_decision_boundary_2d(svm_2d, X2_tr, X2_te, y_tr, y_te, best_C)
