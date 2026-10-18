"""Metrics, decision rules, diagnostic plots and subgroup analysis."""
from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score, fbeta_score,
                             make_scorer, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

from .preprocessing import CLASS_NAMES

CLASS_COLORS = ["#4C72B0", "#DD8452", "#C44E52"]
N_CLASSES = len(CLASS_NAMES)


def macro_average_precision(y_true, proba) -> float:
    """Mean one-vs-rest average precision (area under the PR curve) across classes."""
    Y = label_binarize(y_true, classes=list(range(N_CLASSES)))
    return float(np.mean([average_precision_score(Y[:, k], proba[:, k]) for k in range(N_CLASSES)]))


MACRO_AP_SCORER = make_scorer(macro_average_precision, response_method="predict_proba")


def compute_metrics(y_true, proba, y_pred=None, sample_weight=None) -> dict:
    """Threshold-free and threshold-dependent metrics for 3-class predictions.

    Macro averages weight every class equally, so the rare prediabetes class
    counts as much as the majority class; this is the primary lens because
    accuracy is dominated by the 85% 'no diabetes' class.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    if y_pred is None:
        y_pred = proba.argmax(axis=1)
    Y = label_binarize(y_true, classes=list(range(N_CLASSES)))
    sw = sample_weight
    m = {
        "accuracy": accuracy_score(y_true, y_pred, sample_weight=sw),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred, sample_weight=sw),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0, sample_weight=sw),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0, sample_weight=sw),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0, sample_weight=sw),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0, sample_weight=sw),
        "roc_auc_ovr_macro": roc_auc_score(y_true, proba, multi_class="ovr", average="macro", sample_weight=sw),
        "avg_precision_macro": float(np.mean([average_precision_score(Y[:, k], proba[:, k], sample_weight=sw)
                                              for k in range(N_CLASSES)])),
    }
    rec = recall_score(y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0, sample_weight=sw)
    prec = precision_score(y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0, sample_weight=sw)
    f1 = f1_score(y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0, sample_weight=sw)
    for k, name in enumerate(["none", "prediabetes", "diabetes"]):
        m[f"recall_{name}"] = rec[k]
        m[f"precision_{name}"] = prec[k]
        m[f"f1_{name}"] = f1[k]
        m[f"ap_{name}"] = average_precision_score(Y[:, k], proba[:, k], sample_weight=sw)
        m[f"roc_auc_{name}"] = roc_auc_score(Y[:, k], proba[:, k], sample_weight=sw)
    return {k: float(v) for k, v in m.items()}


def predict_with_weights(proba: np.ndarray, class_weights) -> np.ndarray:
    """Decision rule: argmax of probabilities rescaled by per-class weights."""
    return (np.asarray(proba) * np.asarray(class_weights)[None, :]).argmax(axis=1)


def tune_decision_weights(y_true, proba, grid=None, metric=lambda y, p: f1_score(y, p, average="macro")):
    """Grid-search per-class multipliers on *validation* probabilities.

    Rescaling class probabilities is the multiclass analogue of moving a
    binary threshold away from 0.5. The 'no diabetes' weight is fixed at 1.
    Returns (best_weights, best_score, table_of_all_candidates).
    """
    grid = grid if grid is not None else [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80]
    rows = []
    for w_pre, w_dia in itertools.product(grid, grid):
        w = np.array([1.0, w_pre, w_dia])
        rows.append((w_pre, w_dia, metric(y_true, predict_with_weights(proba, w))))
    table = pd.DataFrame(rows, columns=["w_prediabetes", "w_diabetes", "score"]).sort_values("score", ascending=False)
    best = table.iloc[0]
    return np.array([1.0, best.w_prediabetes, best.w_diabetes]), float(best.score), table.reset_index(drop=True)


def screening_threshold(y_true_binary, score, beta: float = 2.0):
    """Choose the binary threshold that maximises F-beta on validation data.

    beta = 2 weights recall twice as heavily as precision, reflecting that a
    missed case (false negative) is costlier than an unnecessary follow-up
    test in a screening setting.
    """
    precision, recall, thresholds = precision_recall_curve(y_true_binary, score)
    fbeta = (1 + beta**2) * precision * recall / np.clip(beta**2 * precision + recall, 1e-12, None)
    idx = int(np.nanargmax(fbeta[:-1]))
    return float(thresholds[idx]), float(fbeta[idx]), float(precision[idx]), float(recall[idx])


def binary_screening_metrics(y_true_binary, score, threshold: float, sample_weight=None) -> dict:
    pred = (np.asarray(score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_binary, pred, sample_weight=sample_weight).ravel()
    return {
        "threshold": threshold,
        "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
        "ppv": tp / (tp + fp) if (tp + fp) else 0.0, "npv": tn / (tn + fn) if (tn + fn) else 0.0,
        "f2": fbeta_score(y_true_binary, pred, beta=2, sample_weight=sample_weight),
        "flag_rate": (tp + fp) / (tn + fp + fn + tp),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def report_frame(y_true, y_pred) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T.round(3)


# ----------------------------------------------------------------- plotting

def _save(fig, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_confusion_matrices(y_true, y_pred, title: str = "", path=None):
    """Side-by-side raw counts and row-normalised (recall) confusion matrices."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=axes[1], vmin=0, vmax=1,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axes[0].set_title("Counts")
    axes[1].set_title("Row-normalised (recall)")
    for ax in axes:
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_roc_pr_curves(y_true, probas: dict, path=None):
    """One-vs-rest ROC and precision-recall curves for each class and model."""
    Y = label_binarize(y_true, classes=list(range(N_CLASSES)))
    fig, axes = plt.subplots(2, N_CLASSES, figsize=(16, 9))
    styles = ["-", "--", ":", "-."]
    for k, cname in enumerate(CLASS_NAMES):
        for (name, proba), ls in zip(probas.items(), itertools.cycle(styles)):
            fpr, tpr, _ = roc_curve(Y[:, k], proba[:, k])
            auc = roc_auc_score(Y[:, k], proba[:, k])
            axes[0, k].plot(fpr, tpr, ls, label=f"{name} (AUC {auc:.3f})")
            prec, rec, _ = precision_recall_curve(Y[:, k], proba[:, k])
            ap = average_precision_score(Y[:, k], proba[:, k])
            axes[1, k].plot(rec, prec, ls, label=f"{name} (AP {ap:.3f})")
        axes[0, k].plot([0, 1], [0, 1], color="grey", lw=0.8)
        axes[1, k].axhline(Y[:, k].mean(), color="grey", lw=0.8, label=f"prevalence {Y[:, k].mean():.3f}")
        axes[0, k].set(title=f"ROC - {cname} vs rest", xlabel="False positive rate", ylabel="True positive rate")
        axes[1, k].set(title=f"Precision-recall - {cname} vs rest", xlabel="Recall", ylabel="Precision")
        axes[0, k].legend(fontsize=8, loc="lower right")
        axes[1, k].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_learning_curves(histories: dict, path=None):
    """Training vs validation loss and accuracy per Keras run."""
    n = len(histories)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 7), squeeze=False)
    for j, (name, h) in enumerate(histories.items()):
        ep = np.arange(1, len(h["loss"]) + 1)
        axes[0, j].plot(ep, h["loss"], label="train")
        axes[0, j].plot(ep, h["val_loss"], label="validation")
        best = int(np.argmin(h["val_loss"])) + 1
        axes[0, j].axvline(best, color="grey", ls=":", lw=1, label=f"best epoch {best}")
        axes[0, j].set(title=f"{name}\nloss", xlabel="epoch")
        axes[1, j].plot(ep, h["accuracy"], label="train")
        axes[1, j].plot(ep, h["val_accuracy"], label="validation")
        axes[1, j].set(title="accuracy", xlabel="epoch")
        axes[0, j].legend(fontsize=8)
        axes[1, j].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_model_comparison(results: pd.DataFrame, metrics: list[str], label_col: str = "experiment_id", path=None):
    """One dot panel per metric, x-axis zoomed to the observed range.

    Differences between experiments are small, so a shared 0-1 axis would hide
    them; each panel marks the best experiment with a vertical line.
    """
    labels = results[label_col].tolist()
    colors = ["#4C72B0" if p == "traditional ML" else "#C44E52" for p in results.get("paradigm", ["traditional ML"] * len(results))]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 0.32 * len(results) + 1.6), sharey=True)
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        vals = results[metric].to_numpy()
        ax.scatter(vals, range(len(vals)), c=colors, s=36, zorder=3)
        ax.axvline(vals.max(), color="grey", ls=":", lw=1)
        span = vals.max() - vals.min()
        ax.set_xlim(vals.min() - 0.08 * span - 1e-3, vals.max() + 0.08 * span + 1e-3)
        ax.set_title(metric, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks(range(len(labels)))
    ax0.set_yticklabels(labels)
    ax0.invert_yaxis()
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color="#4C72B0", label="traditional ML"),
               Line2D([], [], marker="o", ls="", color="#C44E52", label="deep learning")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Validation metrics by experiment (dotted line = best)", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path)
    return fig


def subgroup_metrics(y_true, proba, y_pred, groups: pd.Series, min_positive: int = 30) -> pd.DataFrame:
    """Per-subgroup recall/precision for the diabetes class and macro scores.

    Subgroups with fewer than ``min_positive`` diabetes cases are reported but
    flagged, because their estimates are too unstable to interpret.
    """
    rows = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for g in pd.Series(groups).dropna().unique():
        mask = (np.asarray(groups) == g)
        yt, yp, pr = y_true[mask], y_pred[mask], proba[mask]
        pos = int((yt == 2).sum())
        row = {"group": g, "n": int(mask.sum()), "diabetes_cases": pos,
               "diabetes_prevalence": float((yt == 2).mean()),
               "recall_diabetes": recall_score(yt == 2, yp == 2, zero_division=0),
               "precision_diabetes": precision_score(yt == 2, yp == 2, zero_division=0),
               "f1_macro": f1_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0),
               "reliable": pos >= min_positive}
        try:
            row["roc_auc_diabetes"] = roc_auc_score(yt == 2, pr[:, 2])
        except ValueError:
            row["roc_auc_diabetes"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
