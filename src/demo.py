import sys
sys.path.append("src")

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import json
import os

from model import SoccerNetTCN
from game_dataset import FEATURE_CONFIG
from dataset import SELECTED_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, BACKGROUND_IDX
from utils import load_checkpoint
from SoccerNet.utils import getListGames


def predict_half(model, npy_path, device, fps=1):
    features = np.load(npy_path).astype(np.float32)
    features_tensor = torch.tensor(
        features, dtype=torch.float32
    ).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(features_tensor)
        probs = torch.softmax(logits, dim=2).squeeze(0).cpu().numpy()

    return probs, features.shape[0]


def find_peaks(probs, cls_idx, min_confidence=0.02,
               min_distance_frames=4):
    class_probs = probs[:, cls_idx]
    peaks = []
    i = 1
    while i < len(class_probs) - 1:
        if (class_probs[i] > class_probs[i-1] and
                class_probs[i] > class_probs[i+1] and
                class_probs[i] >= min_confidence):
            if peaks and (i - peaks[-1][0]) < min_distance_frames:
                if class_probs[i] > peaks[-1][1]:
                    peaks[-1] = (i, class_probs[i])
            else:
                peaks.append((i, float(class_probs[i])))
        i += 1
    return peaks


def load_ground_truth(label_path, half, fps=1):
    with open(label_path, "r") as f:
        data = json.load(f)

    annotations = []
    for ann in data["annotations"]:
        if int(ann["gameTime"].split(" - ")[0]) != half:
            continue
        if ann["label"] not in SELECTED_CLASSES:
            continue
        time_str = ann["gameTime"].split(" - ")[1]
        minutes, seconds = map(int, time_str.split(":"))
        total_seconds = minutes * 60 + seconds
        annotations.append({
            "label": ann["label"],
            "cls_idx": CLASS_TO_IDX[ann["label"]],
            "seconds": total_seconds,
            "frame": int(total_seconds * fps)
        })

    return annotations


def run_demo(
    checkpoint_path,
    data_path,
    game_name,
    half=1,
    min_confidence=0.02,
    tolerance_seconds=2,
    classes_to_show=None,
    save_path=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SoccerNetTCN(
        input_dim=8576,
        d_model=256,
        num_layers=8,
        kernel_size=3,
        dropout=0.1,
        num_classes=18,
        use_input_norm=True
    )
    load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device)
    print(f"Model loaded from {checkpoint_path}")

    fps = FEATURE_CONFIG["baidu"]["fps"]
    npy_file = FEATURE_CONFIG["baidu"]["files"][half - 1]
    npy_path = os.path.join(data_path, game_name, npy_file)
    label_path = os.path.join(data_path, game_name, "Labels-v2.json")

    print(f"\nGame: {game_name}")
    print(f"Half: {half}")

    probs, num_frames = predict_half(model, npy_path, device, fps)
    ground_truth = load_ground_truth(label_path, half, fps)

    duration_seconds = num_frames / fps
    print(f"Duration: {duration_seconds/60:.1f} minutes ({num_frames} frames)")
    print(f"Ground truth annotations: {len(ground_truth)}")

    predictions = {}
    for cls_idx in range(len(SELECTED_CLASSES)):
        peaks = find_peaks(probs, cls_idx, min_confidence)
        if peaks:
            predictions[cls_idx] = peaks

    total_preds = sum(len(p) for p in predictions.values())
    print(f"Total predictions: {total_preds} (min_confidence={min_confidence})")

    matched_gt = set()
    matched_pred = defaultdict(list)
    for cls_idx, peaks in predictions.items():
        for frame_idx, confidence in peaks:
            pred_seconds = frame_idx / fps
            for gt_idx, gt in enumerate(ground_truth):
                if (gt["cls_idx"] == cls_idx and
                        gt_idx not in matched_gt and
                        abs(gt["seconds"] - pred_seconds) <= tolerance_seconds):
                    matched_gt.add(gt_idx)
                    matched_pred[cls_idx].append(
                        (frame_idx, confidence, True)
                    )
                    break
            else:
                matched_pred[cls_idx].append(
                    (frame_idx, confidence, False)
                )

    correct = len(matched_gt)
    total_gt = len(ground_truth)
    recall = correct / total_gt * 100 if total_gt > 0 else 0
    print(f"\nDetected {correct}/{total_gt} ground truth events "
          f"({recall:.1f}% recall at ±{tolerance_seconds}s tolerance)")

    colors = plt.cm.tab20(np.linspace(0, 1, 17))
    class_colors = {i: colors[i] for i in range(17)}

    if classes_to_show is None:
        gt_classes = set(a["cls_idx"] for a in ground_truth)
        pred_classes = set(predictions.keys())
        classes_to_show = sorted(gt_classes | pred_classes)

    fig, axes = plt.subplots(
        len(classes_to_show), 1,
        figsize=(14, max(4, len(classes_to_show) * 0.7)),
        sharex=True
    )
    if len(classes_to_show) == 1:
        axes = [axes]

    game_short = game_name.replace("\\", "/").split("/")[-1]
    fig.suptitle(
        f"Predicții model vs Ground Truth\n"
        f"{game_short} - Repriza {half} "
        f"(min_confidence={min_confidence}, toleranță=±{tolerance_seconds}s)",
        fontsize=12, fontweight="bold"
    )

    duration_minutes = duration_seconds / 60

    for ax_idx, cls_idx in enumerate(classes_to_show):
        ax = axes[ax_idx]
        cls_name = IDX_TO_CLASS.get(cls_idx, f"Class {cls_idx}")
        color = class_colors[cls_idx]

        ax.set_xlim(0, duration_minutes)
        ax.set_ylim(-1, 1)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.set_yticks([])
        ax.set_ylabel(cls_name, fontsize=8, rotation=0,
                      ha="right", va="center", labelpad=120)

        for gt in ground_truth:
            if gt["cls_idx"] != cls_idx:
                continue
            gt_min = gt["seconds"] / 60
            is_detected = ground_truth.index(gt) in matched_gt
            marker = "^" if is_detected else "x"
            gt_color = "green" if is_detected else "red"
            ax.plot(gt_min, 0.5, marker=marker, color=gt_color,
                    markersize=10, zorder=5,
                    markeredgewidth=2 if marker == "x" else 1)

        if cls_idx in predictions:
            for frame_idx, confidence, is_correct in matched_pred.get(
                    cls_idx, []):
                pred_min = frame_idx / fps / 60
                pred_color = "green" if is_correct else "orange"
                ax.plot(pred_min, -0.5, marker="v",
                        color=pred_color, markersize=8,
                        alpha=max(0.3, confidence), zorder=5)
                ax.text(pred_min, -0.85, f"{confidence:.0%}",
                        ha="center", fontsize=6, color=pred_color)

        ax.grid(axis="x", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    axes[-1].set_xlabel("Timp (minute)", fontsize=10)

    legend_elements = [
        mpatches.Patch(color="green", label="Ground truth — detectat ✓"),
        mpatches.Patch(color="red", label="Ground truth — ratat ✗"),
        mpatches.Patch(color="green", alpha=0.6,
                       label="Predicție corectă ▼"),
        mpatches.Patch(color="orange", alpha=0.6,
                       label="Predicție falsă pozitivă ▼"),
    ]
    fig.legend(handles=legend_elements, loc="lower right",
               fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigura salvată la: {save_path}")

    plt.show()
    return probs, ground_truth, predictions


if __name__ == "__main__":
    game_list = getListGames("test")
    print("Primele 5 meciuri din setul de test:")
    for i, g in enumerate(game_list[:5]):
        print(f"  {i}: {g}")

    run_demo(
        checkpoint_path="checkpoints/tcn_baidu/finetune_tcn_baidu_per_class_radius_best.pt",#checkpoint_path="checkpoints/tcn_baidu/finetune_tcn_baidu_pretrained_best.pt", 
        data_path="D:/soccernet-data",
        game_name=game_list[4],
        half=1,
        min_confidence=0.10,
        tolerance_seconds=2,
        classes_to_show=None,
        save_path="results/figures/demo_prediction_per_class_radius_meci_5_conf10.png"
    )