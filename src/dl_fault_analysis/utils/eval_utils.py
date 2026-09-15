from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader


def _binary_logits_to_pred(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    return (torch.sigmoid(logits) >= threshold).to(torch.int64)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_type: str,
    binary_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate model on a data loader.

    Computes loss and task-specific metrics.

    Args:
        model: PyTorch model.
        loader: DataLoader for evaluation.
        device: Device to run on.
        task_type: "binary", "multiclass", or "regression".
        binary_threshold: Threshold for binary classification.

    Returns:
        Dict of metric names to values.
    """
    model.eval()

    y_true_list: List[np.ndarray] = []
    y_pred_list: List[np.ndarray] = []
    y_prob_list: List[np.ndarray] = []  # for AUC metrics
    losses: List[float] = []

    if task_type == "binary":
        criterion = nn.BCEWithLogitsLoss()
    elif task_type == "multiclass":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        out = model(xb)

        if task_type == "multiclass":
            loss = criterion(out, yb)
            pred = torch.argmax(out, dim=-1)
            probs = torch.softmax(out, dim=-1)
            y_true_list.append(yb.cpu().numpy())
            y_pred_list.append(pred.cpu().numpy())
            y_prob_list.append(probs.cpu().numpy())
            losses.append(loss.item())

        elif task_type == "binary":
            logits = out.squeeze(-1)
            loss = criterion(logits, yb)
            pred = _binary_logits_to_pred(logits, threshold=binary_threshold)
            probs = torch.sigmoid(logits)
            y_true_list.append(yb.cpu().numpy())
            y_pred_list.append(pred.cpu().numpy())
            y_prob_list.append(probs.cpu().numpy())
            losses.append(loss.item())

        else:
            pred = out.squeeze(-1)
            loss = criterion(pred, yb)
            y_true_list.append(yb.cpu().numpy())
            y_pred_list.append(pred.cpu().numpy())
            losses.append(loss.item())

    y_true = np.concatenate(y_true_list, axis=0) if y_true_list else np.array([])
    y_pred = np.concatenate(y_pred_list, axis=0) if y_pred_list else np.array([])
    y_prob = np.concatenate(y_prob_list, axis=0) if y_prob_list else np.array([])
    loss_mean = float(np.mean(losses)) if losses else float("nan")

    metrics: Dict[str, float] = {"loss": loss_mean}

    if task_type == "binary":
        metrics["accuracy"] = (
            float(accuracy_score(y_true, y_pred)) if y_true.size else float("nan")
        )
        metrics["f1"] = (
            float(f1_score(y_true, y_pred, average="binary"))
            if y_true.size
            else float("nan")
        )
        # ROC AUC and PR AUC
        if y_true.size > 0 and len(np.unique(y_true)) > 1:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
                metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
            except ValueError:
                metrics["roc_auc"] = float("nan")
                metrics["pr_auc"] = float("nan")
        else:
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"] = float("nan")
    elif task_type == "multiclass":
        metrics["accuracy"] = (
            float(accuracy_score(y_true, y_pred)) if y_true.size else float("nan")
        )
        metrics["f1_macro"] = (
            float(f1_score(y_true, y_pred, average="macro"))
            if y_true.size
            else float("nan")
        )
        # ROC AUC and PR AUC (macro-averaged OvR)
        if y_true.size > 0 and y_prob.size > 0 and len(np.unique(y_true)) > 1:
            try:
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
                # For PR AUC in multiclass, use one-vs-rest with macro averaging
                metrics["pr_auc_macro"] = float(
                    average_precision_score(y_true, y_prob, average="macro")
                )
            except ValueError:
                metrics["roc_auc_ovr"] = float("nan")
                metrics["pr_auc_macro"] = float("nan")
        else:
            metrics["roc_auc_ovr"] = float("nan")
            metrics["pr_auc_macro"] = float("nan")
    else:
        metrics["mae"] = (
            float(mean_absolute_error(y_true, y_pred)) if y_true.size else float("nan")
        )
        metrics["rmse"] = (
            float(np.sqrt(mean_squared_error(y_true, y_pred)))
            if y_true.size
            else float("nan")
        )

    return metrics


def predict_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_type: str,
    binary_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (y_true, y_pred, y_score) as numpy arrays for classification tasks.

    For binary classification:
        y_score: sigmoid probabilities (shape: N,)

    For multiclass classification:
        y_score: softmax probabilities for all classes (shape: N, num_classes)

    For regression:
        y_score: same as y_pred
    """
    y_true_list = []
    y_pred_list = []
    y_score_list = []

    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            out = model(xb)

            if task_type == "regression":
                # shapes: (B,1) -> (B,)
                yp = (
                    out.detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(np.float32, copy=False)
                )
                yt = (
                    yb.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
                )
                ys = yp  # for regression, score is same as prediction

            elif task_type == "binary":
                probs = torch.sigmoid(out).detach().cpu().numpy().reshape(-1)
                yp = (probs >= binary_threshold).astype(np.int64, copy=False)
                yt = yb.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
                ys = probs  # sigmoid probabilities

            elif task_type == "multiclass":
                probs = torch.softmax(out, dim=1).detach().cpu().numpy()
                preds = np.argmax(probs, axis=1).astype(np.int64, copy=False)
                yp = preds
                yt = yb.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
                ys = probs  # full probability matrix (B, num_classes)

            else:
                raise ValueError(f"Unknown task_type={task_type}")

            y_true_list.append(yt)
            y_pred_list.append(yp)
            y_score_list.append(ys)

    y_true_np = np.concatenate(y_true_list, axis=0)
    y_pred_np = np.concatenate(y_pred_list, axis=0)
    y_score_np = np.concatenate(y_score_list, axis=0)
    return y_true_np, y_pred_np, y_score_np
