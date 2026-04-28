import os
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from model import SoccerNetTCN
from game_dataset import FEATURE_CONFIG, SoccerNetGameDataset, get_game_dataloader
from dataset import SELECTED_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, BACKGROUND_IDX
from utils import get_device, load_checkpoint
from SoccerNet.utils import getListGames


def predict_full_half(model, features, device):
    """
    Runs full sequence inference on one match half.
    Returns per-frame class probabilities.

    features : tensor of shape (1, seq_len, 512)
    returns  : tensor of shape (seq_len, num_classes)
    """
    model.eval()
    with torch.no_grad():
        features = features.to(device)
        logits = model(features)
        probs = torch.softmax(logits, dim=2)
    return probs.squeeze(0).cpu()


def find_peaks(probs, cls_idx, min_confidence=0.3,
               min_distance_frames=4):
    """
    Finds local peaks in a class probability timeline.
    Returns list of (frame_idx, confidence) tuples.

    probs             : tensor of shape (seq_len, num_classes)
    cls_idx           : class index to find peaks for
    min_confidence    : minimum probability threshold
    min_distance_frames: minimum gap between peaks
    """
    class_probs = probs[:, cls_idx].numpy()
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
                peaks.append((i, class_probs[i]))
        i += 1

    return peaks


def compute_tcn_average_map(model, data_path, split, device,
                             tolerances=[1, 2, 3, 4, 5],
                             fps=2, min_confidence=0.3, feature_type="baidu"):
    """
    Computes Average-mAP for the TCN dense prediction model.
    Uses peak detection instead of sliding window inference.
    """
    game_list = getListGames(split)
    all_predictions = defaultdict(list)
    all_ground_truths = defaultdict(list)

    print(f"Running dense inference on {len(game_list)} games...")

    for game in tqdm(game_list):
        game_path = os.path.join(data_path, game)
        label_path = os.path.join(game_path, "Labels-v2.json")

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            data = json.load(f)

        for half in [1, 2]:
            from game_dataset import FEATURE_CONFIG
            config = FEATURE_CONFIG[feature_type]
            npy_file = config["files"][half - 1]
            npy_path = os.path.join(game_path, npy_file)

            if not os.path.exists(npy_path):
                continue

            features = np.load(npy_path).astype(np.float32)
            features_tensor = torch.tensor(
                features, dtype=torch.float32
            ).unsqueeze(0)

            probs = predict_full_half(model, features_tensor, device)

            for cls_idx in range(len(SELECTED_CLASSES)):
                peaks = find_peaks(
                    probs, cls_idx,
                    min_confidence=min_confidence
                )
                for frame_idx, confidence in peaks:
                    seconds = frame_idx / fps
                    all_predictions[cls_idx].append({
                        "game": game,
                        "half": half,
                        "seconds": seconds,
                        "confidence": confidence
                    })

            half_annotations = [
                a for a in data["annotations"]
                if int(a["gameTime"].split(" - ")[0]) == half
                and a["label"] in SELECTED_CLASSES
            ]

            for ann in half_annotations:
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
                            abs(gt["seconds"] -
                                pred["seconds"]) <= tolerance and
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
                precision[i] = max(precision[i], precision[i+1])

            ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
            aps.append(ap)
            ap_per_class_per_tolerance[cls_idx][tolerance] = ap

        map_per_tolerance[tolerance] = np.mean(aps) if aps else 0.0

    average_map = np.mean(list(map_per_tolerance.values()))

    return {
        "average_map": average_map,
        "map_per_tolerance": map_per_tolerance,
        "ap_per_class_per_tolerance": ap_per_class_per_tolerance
    }


def evaluate_tcn_per_class(model, dataset, device):
    """
    Evaluates TCN on a game dataset computing per-frame
    classification metrics.
    """
    model.eval()
    all_predictions = []
    all_labels = []

    loader = get_game_dataloader(dataset, shuffle=False)

    with torch.no_grad():
        for features, labels in tqdm(loader, desc="Evaluating"):
            features = features.to(device)
            logits = model(features)
            preds = logits.argmax(dim=2).squeeze(0)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.squeeze(0).numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    class_names = [IDX_TO_CLASS.get(i, "Background")
                   for i in range(len(SELECTED_CLASSES) + 1)]

    report = classification_report(
        all_labels, all_predictions,
        target_names=class_names, digits=4
    )

    overall_acc = (all_predictions == all_labels).mean() * 100

    return {
        "predictions": all_predictions,
        "labels": all_labels,
        "report": report,
        "overall_acc": overall_acc,
        "class_names": class_names
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate TCN model"
    )
    parser.add_argument("--data_path", type=str,
                        default="D:/soccernet-data")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/tcn/finetune_tcn_pretrained_best.pt")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--min_confidence", type=float, default=0.3)
    parser.add_argument("--compute_map", action="store_true")
    parser.add_argument("--feature_type", type=str, default="baidu",
                    choices=["resnet", "baidu"])
    parser.add_argument("--label_radius", type=int, default=2)
    parser.add_argument("--run_name", type=str, default="tcn")
    args = parser.parse_args()

    device = get_device()
    input_dim = FEATURE_CONFIG[args.feature_type]["input_dim"]
    use_input_norm = args.feature_type == "baidu"
    model = SoccerNetTCN(
        input_dim=input_dim,
        d_model=256,
        num_layers=8,
        kernel_size=3,
        dropout=0.1,
        num_classes=18,
        use_input_norm=use_input_norm
    )

    load_checkpoint(args.checkpoint, model, device=device)
    model = model.to(device)

    print(f"\nLoading {args.split} dataset...")
    test_dataset = SoccerNetGameDataset(
        data_path=args.data_path,
        split=args.split,
        feature_type=args.feature_type,
        label_radius=args.label_radius
    )

    print("\n--- Per-frame evaluation ---")
    results = evaluate_tcn_per_class(model, test_dataset, device)
    print(f"\nOverall test accuracy: {results['overall_acc']:.2f}%")
    print("\nClassification report:")
    print(results["report"])

    cm = confusion_matrix(results["labels"], results["predictions"])
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=results["class_names"],
        yticklabels=results["class_names"]
    )
    plt.title("TCN confusion matrix — normalized")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"results/figures/confusion_matrix_{args.run_name}.png",
            dpi=150, bbox_inches="tight")
    plt.show()
    print("Confusion matrix saved.")

    if args.compute_map:
        print("\n--- Average-mAP evaluation ---")
        map_results = compute_tcn_average_map(
            model=model,
            data_path=args.data_path,
            split=args.split,
            device=device,
            min_confidence=args.min_confidence,
            feature_type=args.feature_type,
            fps=FEATURE_CONFIG[args.feature_type]["fps"]
        )

        print("\nAverage-mAP Results:")
        print("-" * 40)
        for tol, map_val in map_results["map_per_tolerance"].items():
            print(f"  mAP @ {tol}s tolerance: {map_val*100:.2f}%")
        print("-" * 40)
        print(f"  Average-mAP: "
              f"{map_results['average_map']*100:.2f}%")

        print("\nPer-class AP (averaged across tolerances):")
        for cls_idx, cls_name in IDX_TO_CLASS.items():
            aps = list(
                map_results["ap_per_class_per_tolerance"]
                [cls_idx].values()
            )
            if aps:
                mean_ap = np.mean(aps) * 100
                print(f"  {cls_name}: {mean_ap:.2f}%")

        np.save(f"results/{args.run_name}_map_results.npy", map_results)
        print(f"\nmAP results saved to results/{args.run_name}_map_results.npy")