import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict

from dataset import SoccerNetDataset, get_dataloader, SELECTED_CLASSES, IDX_TO_CLASS, BACKGROUND_IDX
from model import SoccerNetTransformer


def extract_attention_weights(model, window, device):
    """
    Extracts attention weights from all Transformer encoder layers
    for a single input window.

    model  : trained SoccerNetTransformer
    window : tensor of shape (1, window_size, input_dim)
    device : torch device

    Returns list of attention weight matrices, one per layer,
    each of shape (1, num_heads, window_size, window_size)
    """
    model.eval()
    attention_weights = []

    with torch.no_grad():
        x = model.input_projection(window)
        x = model.positional_encoding(x)

        for layer in model.transformer_encoder.layers:
            x_norm = layer.norm1(x)
            attn_output, attn_weights = layer.self_attn(
                x_norm, x_norm, x_norm,
                need_weights=True,
                average_attn_weights=False
            )
            attention_weights.append(attn_weights.detach().cpu())
            x = x + layer.dropout1(attn_output)
            x = x + layer.dropout2(
                layer.linear2(
                    layer.dropout(
                        torch.nn.functional.relu(
                            layer.linear1(layer.norm2(x))
                        )
                    )
                )
            )

    return attention_weights


def plot_attention_heatmaps(attention_weights, true_label, pred_label,
                             save_path=None):
    """
    Plots attention weight heatmaps for all layers and heads.

    attention_weights : list of tensors, one per layer,
                        each (num_heads, window_size, window_size)
    true_label        : string name of true class
    pred_label        : string name of predicted class
    """
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].shape[1]

    fig, axes = plt.subplots(
        num_layers, num_heads,
        figsize=(num_heads * 4, num_layers * 4)
    )

    for layer_idx in range(num_layers):
        attn = attention_weights[layer_idx].squeeze(0).numpy()
        for head_idx in range(num_heads):
            ax = axes[layer_idx][head_idx]
            sns.heatmap(
                attn[head_idx],
                ax=ax,
                cmap="viridis",
                cbar=False,
                xticklabels=False,
                yticklabels=False
            )
            ax.set_title(
                f"Layer {layer_idx + 1}, Head {head_idx + 1}",
                fontsize=9
            )
            if head_idx == 0:
                ax.set_ylabel(f"Layer {layer_idx + 1}", fontsize=9)

    correct = "Correct" if true_label == pred_label else "Wrong"
    fig.suptitle(
        f"Attention weights — True: {true_label} | "
        f"Predicted: {pred_label} ({correct})",
        fontsize=12
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Attention heatmap saved to {save_path}")

    plt.show()


def plot_confidence_calibration(model, dataloader, device, 
                                 num_bins=10, save_path=None):
    """
    Plots a confidence calibration curve.
    Shows whether predicted confidence matches actual accuracy.
    A perfectly calibrated model follows the diagonal line.
    """
    model.eval()
    all_confidences = []
    all_correct = []

    with torch.no_grad():
        for batch_windows, batch_labels in tqdm(
            dataloader, desc="Computing calibration"
        ):
            batch_windows = batch_windows.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_windows)
            probs = torch.softmax(logits, dim=1)
            confidences, predictions = probs.max(dim=1)

            correct = (predictions == batch_labels).float()
            all_confidences.extend(confidences.cpu().numpy())
            all_correct.extend(correct.cpu().numpy())

    all_confidences = np.array(all_confidences)
    all_correct = np.array(all_correct)

    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(num_bins):
        mask = (all_confidences >= bin_edges[i]) & \
               (all_confidences < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accs.append(all_correct[mask].mean())
            bin_confs.append(all_confidences[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accs.append(0)
            bin_confs.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_counts.append(0)

    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)

    ece = np.sum(
        np.array(bin_counts) / len(all_confidences) *
        np.abs(bin_accs - bin_confs)
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.plot(bin_confs, bin_accs, "o-", color="steelblue",
             linewidth=2, markersize=6, label="Model calibration")
    ax1.fill_between(bin_confs, bin_accs, bin_confs,
                     alpha=0.2, color="red", label="Calibration gap")
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Calibration curve (ECE: {ece:.4f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.bar(bin_confs, bin_counts, width=0.08,
            color="steelblue", alpha=0.7)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Sample count")
    ax2.set_title("Confidence distribution")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Model confidence calibration", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Calibration plot saved to {save_path}")

    plt.show()
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    return ece


def plot_error_breakdown(predictions, labels, class_names, save_path=None):
    """
    For each class, shows what the model predicts when it's wrong.
    Reveals systematic confusion patterns beyond the confusion matrix.
    """
    num_classes = len(class_names)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for cls_idx in range(num_classes):
        ax = axes[cls_idx]
        true_mask = labels == cls_idx
        wrong_mask = true_mask & (predictions != labels)

        if wrong_mask.sum() == 0:
            ax.text(0.5, 0.5, "No errors",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{class_names[cls_idx]}")
            continue

        wrong_predictions = predictions[wrong_mask]
        wrong_counts = np.bincount(wrong_predictions, minlength=num_classes)
        wrong_counts[cls_idx] = 0

        colors = ["steelblue" if i != cls_idx else "white"
                  for i in range(num_classes)]
        ax.barh(class_names, wrong_counts, color=colors)
        ax.set_title(
            f"{class_names[cls_idx]}\n"
            f"({wrong_mask.sum()} errors / {true_mask.sum()} total)"
        )
        ax.set_xlabel("Error count")
        ax.invert_yaxis()

    axes[-1].axis("off")
    plt.suptitle("Error breakdown — what does the model predict when wrong?",
                 fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Error breakdown saved to {save_path}")

    plt.show()


def plot_confidence_by_class(model, dataloader, device, save_path=None):
    """
    Plots the distribution of model confidence scores per class,
    split by correct and incorrect predictions.
    Reveals whether the model is overconfident on wrong predictions.
    """
    model.eval()
    class_confidences_correct = defaultdict(list)
    class_confidences_wrong = defaultdict(list)

    with torch.no_grad():
        for batch_windows, batch_labels in tqdm(
            dataloader, desc="Analyzing confidence"
        ):
            batch_windows = batch_windows.to(device)
            logits = model(batch_windows)
            probs = torch.softmax(logits, dim=1)
            confidences, predictions = probs.max(dim=1)

            for i in range(len(batch_labels)):
                true_cls = batch_labels[i].item()
                pred_cls = predictions[i].item()
                conf = confidences[i].item()
                cls_name = IDX_TO_CLASS.get(true_cls, "Background")

                if pred_cls == true_cls:
                    class_confidences_correct[cls_name].append(conf)
                else:
                    class_confidences_wrong[cls_name].append(conf)

    class_names = [IDX_TO_CLASS.get(i, "Background")
                   for i in range(len(SELECTED_CLASSES) + 1)]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, cls_name in enumerate(class_names):
        ax = axes[idx]
        correct_confs = class_confidences_correct[cls_name]
        wrong_confs = class_confidences_wrong[cls_name]

        if correct_confs:
            ax.hist(correct_confs, bins=20, alpha=0.6,
                    color="steelblue", label="Correct", density=True)
        if wrong_confs:
            ax.hist(wrong_confs, bins=20, alpha=0.6,
                    color="red", label="Wrong", density=True)

        ax.set_title(cls_name)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].axis("off")
    plt.suptitle(
        "Confidence distribution — correct vs wrong predictions per class",
        fontsize=13
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confidence plot saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    import argparse
    from evaluate import evaluate_per_class

    parser = argparse.ArgumentParser(description="Analysis and visualization")
    parser.add_argument("--data_path", type=str, default="D:/soccernet-data")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/finetune_best.pt")
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--attention", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SoccerNetTransformer(
        input_dim=512,
        d_model=256,
        num_heads=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        num_classes=7
    )

    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    print("\nLoading test dataset...")
    test_dataset = SoccerNetDataset(
        data_path=args.data_path,
        split="test",
        window_size=args.window_size,
        overlap=0.5
    )
    test_loader = get_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    print("\n--- Running per-class evaluation ---")
    results = evaluate_per_class(model, test_loader, device)

    print("\n--- Confidence calibration ---")
    ece = plot_confidence_calibration(
        model, test_loader, device,
        save_path="results/figures/calibration.png"
    )

    print("\n--- Error breakdown ---")
    class_names = [IDX_TO_CLASS.get(i, "Background")
                   for i in range(len(SELECTED_CLASSES) + 1)]
    plot_error_breakdown(
        results["predictions"],
        results["labels"],
        class_names,
        save_path="results/figures/error_breakdown.png"
    )

    print("\n--- Confidence by class ---")
    plot_confidence_by_class(
        model, test_loader, device,
        save_path="results/figures/confidence_by_class.png"
    )

    if args.attention:
        print("\n--- Attention visualization ---")
        os.makedirs("results/figures/attention", exist_ok=True)

        classes_to_visualize = {**IDX_TO_CLASS, BACKGROUND_IDX: "Background"}
        for cls_idx, cls_name in classes_to_visualize.items():

            for window, label in test_dataset.samples:
                if label == cls_idx:
                    window_tensor = torch.tensor(
                        window, dtype=torch.float32
                    ).unsqueeze(0).to(device)

                    logits = model(window_tensor)
                    pred_idx = logits.argmax(dim=1).item()
                    pred_name = IDX_TO_CLASS.get(pred_idx, "Background")

                    attn_weights = extract_attention_weights(
                        model, window_tensor, device
                    )
                    plot_attention_heatmaps(
                        attn_weights,
                        true_label=cls_name,
                        pred_label=pred_name,
                        save_path=f"results/figures/attention/"
                                  f"attention_{cls_name.replace(' ', '_')}.png"
                    )
                    break

    print("\nAnalysis complete.")
    print("All figures saved to results/figures/")