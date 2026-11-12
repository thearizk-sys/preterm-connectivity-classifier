"""Preterm vs. Term Brain Connectivity Classifier using PCA and Logistic Regression."""
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import GridSearchCV, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def generate_synthetic_cohort(n_subjects: int = 120, n_regions: int = 90, seed: int = 42):
    """Generate symmetric zero-diagonal connectivity matrices and imbalanced clinical metadata."""
    rng = np.random.default_rng(seed)

    # Simulate symmetric brain connectivity matrices with a zero diagonal
    raw = rng.uniform(-1, 1, size=(n_subjects, n_regions, n_regions))
    matrices = 0.5 * (raw + np.transpose(raw, (0, 2, 1)))
    for i in range(n_subjects):
        np.fill_diagonal(matrices[i], 0.0)

    # Simulate imbalanced term (0) vs. preterm (1) labels and scan age
    subject_info = pd.DataFrame({
        "subject_id": [f"sub-{i:03d}" for i in range(n_subjects)],
        "age": rng.normal(loc=40.0, scale=2.5, size=n_subjects),
        "prematurity": rng.choice([0, 1], size=n_subjects, p=[0.8, 0.2]),
    })

    region_labels = pd.DataFrame({
        "region_id": np.arange(1, n_regions + 1),
        "region_name": [f"ROI_{i:02d}" for i in range(1, n_regions + 1)],
        "network": rng.choice(
            ["Frontal", "Parietal", "Temporal", "Occipital", "Subcortical"],
            size=n_regions,
        ),
    })

    return matrices, subject_info, region_labels


def extract_upper_triangle(matrices: np.ndarray) -> np.ndarray:
    """Vectorise strict upper triangles of symmetric connectivity matrices."""
    n_subjects, n_regions, _ = matrices.shape
    n_edges = round(n_regions * (n_regions - 1) / 2)
    features = np.zeros((n_subjects, n_edges))

    for i in range(n_subjects):
        idx = 0
        for j in range(n_regions):
            for k in range(j):
                features[i, idx] = matrices[i, j, k]
                idx += 1
    return features


def evaluate_train(model, X_train: np.ndarray, y_train: np.ndarray) -> None:
    """Fit model and print training accuracy."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    acc = accuracy_score(y_train, y_pred)
    print(f"Training Accuracy:    {acc:.4f}")


def evaluate_cv(model, X_train: np.ndarray, y_train: np.ndarray, cv: int = 5) -> None:
    """Evaluate sensitivity, specificity, and macro recall via cross-validation."""
    y_pred = cross_val_predict(model, X_train, y_train, cv=cv)
    _ = confusion_matrix(y_train, y_pred)
    sensitivity = recall_score(y_train, y_pred, pos_label=1)
    specificity = recall_score(y_train, y_pred, pos_label=0)
    mean_recall = recall_score(y_train, y_pred, average="macro")
    print(f"CV Sensitivity:       {sensitivity:.4f}")
    print(f"CV Specificity:       {specificity:.4f}")
    print(f"CV Mean Recall:       {mean_recall:.4f}")


def evaluate_test(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Evaluate accuracy, sensitivity, specificity, and macro recall on held-out test data."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred, pos_label=1)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    mean_recall = recall_score(y_test, y_pred, average="macro")
    print(f"Test Accuracy:        {acc:.4f}")
    print(f"Test Sensitivity:     {sensitivity:.4f}")
    print(f"Test Specificity:     {specificity:.4f}")
    print(f"Test Mean Recall:     {mean_recall:.4f}")


def plot_pca_predictions(
    X_pca: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str,
    output_path: Path | None = None,
) -> None:
    """Plot true vs. predicted class labels in 2D PCA space."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap="coolwarm", edgecolor="k", alpha=0.8)
    ax1.set_title(f"{split_name}: True Labels")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")

    ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap="coolwarm", edgecolor="k", alpha=0.8)
    ax2.set_title(f"{split_name}: Predicted Labels")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Execute the end-to-end connectivity classification pipeline on synthetic cohort data."""
    matrices, subject_info, _ = generate_synthetic_cohort(n_subjects=120, n_regions=90, seed=42)
    X = extract_upper_triangle(matrices)
    y = subject_info["prematurity"].to_numpy()

    n_samples, n_features = X.shape
    n_term = int(np.sum(y == 0))
    n_preterm = int(np.sum(y == 1))
    print(f"Samples: {n_samples} | Connectivity Features: {n_features}")
    print(f"Term: {n_term} | Preterm: {n_preterm} | Preterm Proportion: {n_preterm / n_samples:.3f}")

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit scaler and PCA strictly on training data to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # 1. Baseline Logistic Regression
    print("\n--- Baseline Classifier ---")
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    evaluate_train(baseline_model, X_train_pca, y_train)
    evaluate_cv(baseline_model, X_train_pca, y_train)
    evaluate_test(baseline_model, X_test_pca, y_test)

    # 2. L1-Regularised Logistic Regression tuned for Macro Recall
    print("\n--- L1-Regularised Classifier (Macro Recall Optimised) ---")
    param_grid = {"C": np.logspace(-3, 3, 7)}
    base_l1 = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000, random_state=42)
    grid_search = GridSearchCV(base_l1, param_grid, cv=5, scoring="recall_macro")
    grid_search.fit(X_train_pca, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    evaluate_train(best_model, X_train_pca, y_train)
    evaluate_cv(best_model, X_train_pca, y_train)
    evaluate_test(best_model, X_test_pca, y_test)

    # Save PCA prediction visualisations
    plot_pca_predictions(X_train_pca, y_train, best_model.predict(X_train_pca), "Train", Path("pca_train.png"))
    plot_pca_predictions(X_test_pca, y_test, best_model.predict(X_test_pca), "Test", Path("pca_test.png"))


if __name__ == "__main__":
    main()
