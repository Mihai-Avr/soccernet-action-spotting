import os
import json
import torch
import numpy as np
from utils import get_device, load_checkpoint, set_seed
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import SoccerNetDataset, get_dataloader, SELECTED_CLASSES, BACKGROUND_IDX, IDX_TO_CLASS
from model import SoccerNetTransformer
from SoccerNet.utils import getListGames


def evaluate_per_class(model, dataloader, device, num_classes=7):
    """
    Evaluates model on a dataset split computing per-class
    precision, recall, F1 and overall accuracy.

    Returns predictions, true labels and a full classification report.
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_windows, batch_labels in tqdm(dataloader, desc="Evaluating"):
            batch_windows = batch_windows.to(device)
            logits = model(batch_windows)
            predictions = logits.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    class_names = [IDX_TO_CLASS.get(i, "Background") for i in range(num_classes)]

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average=None,
        labels=list(range(num_classes))
    )

    overall_acc = (all_predictions == all_labels).mean() * 100

    return {
        "predictions": all_predictions,
        "labels": all_labels,
        "report": report,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "overall_acc": overall_acc,
        "class_names": class_names
    }


def plot_confusion_matrix(predictions, labels, class_names, save_path=None):
    """
    Plots and optionally saves a normalized confusion matrix.
    """
    cm = confusion_matrix(labels, predictions)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax1
    )
    ax1.set_title("Confusion matrix — raw counts")
    ax1.set_ylabel("True label")
    ax1.set_xlabel("Predicted label")
    ax1.tick_params(axis="x", rotation=45)

    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax2
    )
    ax2.set_title("Confusion matrix — normalized")
    ax2.set_ylabel("True label")
    ax2.set_xlabel("Predicted label")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


def plot_per_class_f1(f1_scores, class_names, save_path=None):
    """
    Plots per-class F1 scores as a horizontal bar chart.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["steelblue" if f >= 0.7 else "darkorange" 
              if f >= 0.5 else "red" for f in f1_scores]

    bars = ax.barh(class_names, f1_scores, color=colors)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-class F1 scores on test set")
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()
    ax.axvline(x=0.7, color="green", linestyle="--", 
               alpha=0.5, label="0.7 threshold")
    ax.axvline(x=0.5, color="orange", linestyle="--", 
               alpha=0.5, label="0.5 threshold")

    for bar, score in zip(bars, f1_scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                f"{score:.3f}", va="center", fontsize=10)

    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"F1 plot saved to {save_path}")

    plt.show()



def sliding_window_inference(model, features, window_size, device, step=1):
    """
    Runs sliding window inference across a full match half.
    Returns predicted class indices and confidence scores per window.

    model       : trained SoccerNetTransformer
    features    : numpy array of shape (num_frames, 512)
    window_size : number of frames per window
    device      : torch device
    step        : step size between windows in frames (1 = dense)
    """
    model.eval()
    num_frames = features.shape[0]
    predictions = []

    with torch.no_grad():
        for start in range(0, num_frames - window_size, step):
            end = start + window_size
            window = features[start:end]
            window_tensor = torch.tensor(
                window, dtype=torch.float32
            ).unsqueeze(0).to(device)

            logits = model(window_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred_class = probs.argmax().item()
            confidence = probs[pred_class].item()
            center_frame = (start + end) // 2
            center_seconds = center_frame / 2.0

            predictions.append({
                "center_seconds": center_seconds,
                "pred_class": pred_class,
                "confidence": confidence,
                "probs": probs.cpu().numpy()
            })

    return predictions


def nms(predictions, window_seconds=10):
    """
    Non-Maximum Suppression — removes duplicate detections.
    Keeps only the most confident prediction within each time window.

    predictions   : list of dicts from sliding_window_inference
    window_seconds: minimum gap between kept predictions in seconds
    """
    action_predictions = [
        p for p in predictions if p["pred_class"] != BACKGROUND_IDX
    ]

    action_predictions.sort(key=lambda x: x["confidence"], reverse=True)

    kept = []
    suppressed = set()

    for i, pred in enumerate(action_predictions):
        if i in suppressed:
            continue
        kept.append(pred)
        for j, other in enumerate(action_predictions):
            if j != i and j not in suppressed:
                if (pred["pred_class"] == other["pred_class"] and
                        abs(pred["center_seconds"] -
                            other["center_seconds"]) < window_seconds):
                    suppressed.add(j)

    return kept


def compute_average_map(model, data_path, split, window_size,
                         device, tolerances=[1, 2, 3, 4, 5], step=2):
    """
    Computes Average-mAP at multiple time tolerances.
    This is the official SoccerNet action spotting metric.

    tolerances : list of time tolerances in seconds
    step       : sliding window step in frames (2 = 1 second at 2fps)
    """
    game_list = getListGames(split)
    all_predictions = defaultdict(list)
    all_ground_truths = defaultdict(list)

    print(f"Running sliding window inference on {len(game_list)} games...")

    for game in tqdm(game_list):
        game_path = os.path.join(data_path, game)
        label_path = os.path.join(game_path, "Labels-v2.json")

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            data = json.load(f)

        for half in [1, 2]:
            npy_path = os.path.join(
                game_path, f"{half}_ResNET_TF2_PCA512.npy"
            )
            if not os.path.exists(npy_path):
                continue

            features = np.load(npy_path).astype(np.float32)

            raw_predictions = sliding_window_inference(
                model, features, window_size, device, step=step
            )
            kept_predictions = nms(raw_predictions, window_seconds=10)

            for pred in kept_predictions:
                cls = pred["pred_class"]
                all_predictions[cls].append({
                    "game": game,
                    "half": half,
                    "seconds": pred["center_seconds"],
                    "confidence": pred["confidence"]
                })

            half_annotations = [
                a for a in data["annotations"]
                if int(a["gameTime"].split(" - ")[0]) == half
                and a["label"] in SELECTED_CLASSES
            ]

            for ann in half_annotations:
                from dataset import CLASS_TO_IDX
                time_str = ann["gameTime"].split(" - ")[1]
                minutes, seconds = map(int, time_str.split(":"))
                total_seconds = minutes * 60 + seconds
                cls_idx = CLASS_TO_IDX[ann["label"]]
                all_ground_truths[cls_idx].append({
                    "game": game,
                    "half": half,
                    "seconds": total_seconds
                })

    print("Computing Average-mAP...")
    ap_per_class_per_tolerance = defaultdict(dict)
    map_per_tolerance = {}

    for tolerance in tolerances:
        aps = []

        for cls_idx in range(len(SELECTED_CLASSES)):
            preds = sorted(
                all_predictions[cls_idx],
                key=lambda x: x["confidence"],
                reverse=True
            )
            gts = all_ground_truths[cls_idx]

            if len(gts) == 0:
                continue

            matched_gts = set()
            tp = []
            fp = []

            for pred in preds:
                matched = False
                for gt_idx, gt in enumerate(gts):
                    if (gt["game"] == pred["game"] and
                            gt["half"] == pred["half"] and
                            abs(gt["seconds"] - pred["seconds"]) <= tolerance and
                            gt_idx not in matched_gts):
                        matched = True
                        matched_gts.add(gt_idx)
                        break

                tp.append(1 if matched else 0)
                fp.append(0 if matched else 1)

            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            recall = tp_cumsum / (len(gts) + 1e-6)

            recall = np.concatenate([[0], recall, [1]])
            precision = np.concatenate([[1], precision, [0]])

            for i in range(len(precision) - 2, -1, -1):
                precision[i] = max(precision[i], precision[i + 1])

            ap = np.sum(
                (recall[1:] - recall[:-1]) * precision[1:]
            )
            aps.append(ap)
            ap_per_class_per_tolerance[cls_idx][tolerance] = ap

        map_per_tolerance[tolerance] = np.mean(aps) if aps else 0.0

    average_map = np.mean(list(map_per_tolerance.values()))

    return {
        "average_map": average_map,
        "map_per_tolerance": map_per_tolerance,
        "ap_per_class_per_tolerance": ap_per_class_per_tolerance
    }


def print_map_results(map_results):
    """
    Prints Average-mAP results in a clean formatted table.
    """
    print("\nAverage-mAP Results:")
    print("-" * 40)
    for tolerance, map_val in map_results["map_per_tolerance"].items():
        print(f"  mAP @ {tolerance}s tolerance: {map_val * 100:.2f}%")
    print("-" * 40)
    print(f"  Average-mAP: {map_results['average_map'] * 100:.2f}%")
    print("\nPer-class AP (averaged across tolerances):")
    for cls_idx, cls_name in IDX_TO_CLASS.items():
        aps = list(
            map_results["ap_per_class_per_tolerance"][cls_idx].values()
        )
        if aps:
            mean_ap = np.mean(aps) * 100
            print(f"  {cls_name}: {mean_ap:.2f}%")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SoccerNet model")
    parser.add_argument("--data_path", type=str, default="D:/soccernet-data")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/finetune_best.pt")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--compute_map", action="store_true")
    parser.add_argument("--map_step", type=int, default=2)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    model = SoccerNetTransformer(
        input_dim=512,
        d_model=256,
        num_heads=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        num_classes=7
    )

    load_checkpoint(args.checkpoint, model, device=device)
    model = model.to(device)

    print(f"\nLoading {args.split} dataset...")
    test_dataset = SoccerNetDataset(
        data_path=args.data_path,
        split=args.split,
        window_size=args.window_size,
        overlap=0.5
    )
    test_loader = get_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    print("\n--- Per-class evaluation ---")
    results = evaluate_per_class(model, test_loader, device)

    print(f"\nOverall test accuracy: {results['overall_acc']:.2f}%")
    print("\nClassification report:")
    print(results["report"])

    plot_confusion_matrix(
        results["predictions"],
        results["labels"],
        results["class_names"],
        save_path="results/figures/confusion_matrix.png"
    )

    plot_per_class_f1(
        results["f1"],
        results["class_names"],
        save_path="results/figures/per_class_f1.png"
    )

    if args.compute_map:
        print("\n--- Average-mAP evaluation ---")
        map_results = compute_average_map(
            model=model,
            data_path=args.data_path,
            split=args.split,
            window_size=args.window_size,
            device=device,
            step=args.map_step
        )
        print_map_results(map_results)

        np.save(
            "results/map_results.npy",
            map_results
        )
        print("\nmAP results saved to results/map_results.npy")