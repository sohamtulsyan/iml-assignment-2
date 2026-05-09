import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datasets import load_dataset


#Loading and preprocessing stuff. 
def load_fashion_mnist(val_split=0.1, seed=42):
    print("Loading Fashion-MNIST …")
    ds = load_dataset("fashion_mnist")

    train_images = np.array(ds["train"]["image"], dtype=np.float32) / 255.0   # (60000, 28, 28)
    train_labels = np.array(ds["train"]["label"])
    test_images  = np.array(ds["test"]["image"],  dtype=np.float32) / 255.0   # (10000, 28, 28)

    # Flatten: (N, 784)
    train_images = train_images.reshape(len(train_images), -1)
    test_images  = test_images.reshape(len(test_images), -1)

    # Train / validation split
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_images))
    n_val = int(len(train_images) * val_split)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train = train_images[train_idx]
    X_val   = train_images[val_idx]
    X_test  = test_images

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test

#Defining many activation functions that we'll use for numerous purposes. 
def relu(x):
    return np.maximum(0.0, x)

def relu_deriv(x):
    return (x > 0).astype(np.float32)

def sigmoid(x):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1.0 - s)



class Autoencoder:
    """
    Autoencoder using MLP. 
    Hidden layers use ReLU; output layer uses Sigmoid.
    """

    def __init__(self, encoder_dims, lr=1e-3, seed=0):
        self.dims = encoder_dims                     
        self.lr   = lr
        rng = np.random.default_rng(seed)

        # Build full layer sequence: encoder + decoder (mirror, skip repeated bottleneck)
        full_dims = encoder_dims + encoder_dims[-2::-1]  # [784,512,64,512,784]
        self.layer_dims = full_dims
        self.n_layers   = len(full_dims) - 1

        self.W, self.b = [], []
        for i in range(self.n_layers):
            fan_in  = full_dims[i]
            fan_out = full_dims[i + 1]
            if i < self.n_layers - 1:          # ReLU layers
                std = np.sqrt(2.0 / fan_in)
            else:                              # Sigmoid output
                std = np.sqrt(1.0 / fan_in)
            self.W.append(rng.standard_normal((fan_in, fan_out)).astype(np.float32) * std)
            self.b.append(np.zeros(fan_out, dtype=np.float32))

    def forward(self, X):
        Z_cache, A_cache = [], [X]
        A = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = A @ W + b
            Z_cache.append(Z)
            if i < self.n_layers - 1:
                A = relu(Z)
            else:
                A = sigmoid(Z)
            A_cache.append(A)
        return A_cache[-1], Z_cache, A_cache

    # Normal MSE
    @staticmethod
    def mse(X, X_hat):
        return float(np.mean((X - X_hat) ** 2))

    #Backprop using the standard definition
    def backward(self, X, Z_cache, A_cache):
        m   = X.shape[0]
        dA  = (A_cache[-1] - X) * (2.0 / m)   # dL/dA_out

        dW_list, db_list = [], []
        for i in reversed(range(self.n_layers)):
            Z  = Z_cache[i]
            A_prev = A_cache[i]
            if i < self.n_layers - 1:
                dZ = dA * relu_deriv(Z)
            else:
                dZ = dA * sigmoid_deriv(Z)
            dW = A_prev.T @ dZ
            db = dZ.sum(axis=0)
            dA = dZ @ self.W[i].T
            dW_list.insert(0, dW)
            db_list.insert(0, db)
        return dW_list, db_list

    #SGD update
    def update(self, dW_list, db_list):
        for i in range(self.n_layers):
            self.W[i] -= self.lr * dW_list[i]
            self.b[i] -= self.lr * db_list[i]

    #Mini Batch
    def fit(self, X_train, X_val,
            epochs=30, batch_size=256, verbose=True):

        n = X_train.shape[0]
        train_losses, val_losses = [], []
        rng = np.random.default_rng(42)

        for ep in range(1, epochs + 1):
            idx = rng.permutation(n)
            ep_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                batch = X_train[idx[start:start + batch_size]]
                X_hat, Z_cache, A_cache = self.forward(batch)
                ep_loss += self.mse(batch, X_hat)
                dW, db = self.backward(batch, Z_cache, A_cache)
                self.update(dW, db)
                n_batches += 1

            train_loss = ep_loss / n_batches
            val_loss   = self.mse(X_val, self.forward(X_val)[0])
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if verbose and (ep % 5 == 0 or ep == 1):
                print(f"    Epoch {ep:3d}/{epochs}  "
                      f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        return train_losses, val_losses


#Experimenting with multiple network sizes

CONFIGS = [
    ("Tiny  [784→64]",   [784, 64]),
    ("Small [784→128→32]", [784, 128, 32]),
    ("Med   [784→256→64]", [784, 256, 64]),
    ("Large [784→512→128→32]", [784, 512, 128, 32]),
    ("XL    [784→1024→256→64]", [784, 1024, 256, 64]),
]

EPOCHS     = 40
BATCH_SIZE = 256
LR         = 5e-3


def run_experiments(X_train, X_val, X_test):
    results = {}
    for label, enc_dims in CONFIGS:
        print(f"\n{'─'*50}")
        print(f"Config: {label}")
        n_params = param_count(enc_dims)
        print(f"  Parameters: {n_params:,}")

        ae = Autoencoder(enc_dims, lr=LR)
        t0 = time.perf_counter()
        train_losses, val_losses = ae.fit(X_train, X_val,
                                          epochs=EPOCHS,
                                          batch_size=BATCH_SIZE)
        elapsed = time.perf_counter() - t0

        test_loss = ae.mse(X_test, ae.forward(X_test)[0])
        print(f"  Test MSE : {test_loss:.5f}   Time: {elapsed:.1f}s")

        results[label] = {
            "train_losses": train_losses,
            "val_losses":   val_losses,
            "test_loss":    test_loss,
            "time":         elapsed,
            "n_params":     n_params,
            "ae":           ae,
            "enc_dims":     enc_dims,
        }
    return results


def param_count(enc_dims):
    full = enc_dims + enc_dims[-2::-1]
    total = 0
    for i in range(len(full) - 1):
        total += full[i] * full[i+1] + full[i+1]
    return total


#Visualisations
def plot_loss_curves(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(results)))

    for ax, loss_key, title in zip(
            axes, ["train_losses", "val_losses"],
            ["Training Loss (MSE)", "Validation Loss (MSE)"]):
        for (label, res), c in zip(results.items(), colors):
            ax.plot(range(1, EPOCHS + 1), res[loss_key],
                    label=label, color=c, linewidth=1.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Autoencoder – Fashion-MNIST: Loss Curves by Network Size",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=150, bbox_inches="tight")
    print("Saved: loss_curves.png")
    plt.show()


def plot_summary_bar(results):
    labels     = list(results.keys())
    test_losses = [r["test_loss"] for r in results.values()]
    times       = [r["time"]      for r in results.values()]
    n_params    = [r["n_params"]  for r in results.values()]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))

    for ax, vals, ylabel, title in zip(
            axes,
            [test_losses, times, n_params],
            ["MSE", "Seconds", "# Parameters"],
            ["Test Reconstruction Loss", "Training Time (s)", "Model Size (#params)"]):
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([l.split("[")[0].strip() for l in labels],
                           rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{v:.4f}" if isinstance(v, float) else f"{v:,}",
                    ha="center", va="bottom", fontsize=8)

    plt.suptitle("Autoencoder Comparison Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("summary_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: summary_comparison.png")
    plt.show()


def plot_reconstructions(results, X_test, n_samples=8):
    """Show original vs reconstructed images for each config."""
    n_configs = len(results)
    fig, axes = plt.subplots(n_configs * 2, n_samples,
                             figsize=(n_samples * 1.5, n_configs * 3))

    rng = np.random.default_rng(7)
    sample_idx = rng.integers(0, len(X_test), n_samples)
    samples = X_test[sample_idx]

    for row_pair, (label, res) in enumerate(results.items()):
        ae   = res["ae"]
        X_hat = ae.forward(samples)[0]

        row_orig  = row_pair * 2
        row_recon = row_pair * 2 + 1

        for col in range(n_samples):
            # Original
            axes[row_orig, col].imshow(
                samples[col].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            axes[row_orig, col].axis("off")
            if col == 0:
                axes[row_orig, col].set_ylabel("Original", fontsize=7)

            # Reconstructed
            axes[row_recon, col].imshow(
                X_hat[col].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            axes[row_recon, col].axis("off")
            if col == 0:
                axes[row_recon, col].set_ylabel(
                    label.split("[")[0].strip() + "\nRecon", fontsize=7)

    plt.suptitle("Original vs Reconstructed – Fashion-MNIST", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("reconstructions.png", dpi=150, bbox_inches="tight")
    print("Saved: reconstructions.png")
    plt.show()


def print_results_table(results):
    print("\n" + "═" * 72)
    print(f"{'Config':<30} {'Params':>10} {'Train MSE':>10} {'Val MSE':>10} {'Test MSE':>10} {'Time(s)':>8}")
    print("─" * 72)
    for label, res in results.items():
        print(f"{label:<30} {res['n_params']:>10,} "
              f"{res['train_losses'][-1]:>10.5f} "
              f"{res['val_losses'][-1]:>10.5f} "
              f"{res['test_loss']:>10.5f} "
              f"{res['time']:>8.1f}")
    print("═" * 72)


#main function to call everything. 
if __name__ == "__main__":
    np.random.seed(42)

    X_train, X_val, X_test = load_fashion_mnist(val_split=0.1)

    results = run_experiments(X_train, X_val, X_test)

    print_results_table(results)
    plot_loss_curves(results)
    plot_summary_bar(results)
    plot_reconstructions(results, X_test)
