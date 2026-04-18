import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def save_checkpoint(state, path):
    """
    Saves a checkpoint dictionary to disk.

    state : dict containing model_state_dict and any other info
    path  : full file path to save to
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, device=None):
    """
    Loads a checkpoint from disk into a model and optionally an optimizer.

    path      : full file path to load from
    model     : SoccerNetTransformer instance
    optimizer : optional optimizer to restore state
    device    : torch device

    Returns the checkpoint dictionary for accessing metadata like epoch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded checkpoint from {path}")
    print(f"  Epoch     : {checkpoint.get('epoch', 'unknown')}")
    if "val_acc" in checkpoint:
        print(f"  Val acc   : {checkpoint['val_acc']:.2f}%")
    if "val_loss" in checkpoint:
        print(f"  Val loss  : {checkpoint['val_loss']:.4f}")
    if "loss" in checkpoint:
        print(f"  Loss      : {checkpoint['loss']:.4f}")

    return checkpoint


def get_device():
    """
    Returns the best available device (CUDA if available, else CPU).
    Prints which device is being used.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    return device


def count_parameters(model):
    """
    Counts and prints total and trainable parameters in a model.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    return total, trainable


def plot_training_curves(history, title="Training curves", save_path=None):
    """
    Plots training and validation loss and accuracy curves.

    history   : dict with keys train_loss, val_loss, train_acc, val_acc
    title     : plot title
    save_path : optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], color="steelblue",
             linewidth=2, label="Train loss")
    ax1.plot(epochs, history["val_loss"], color="darkorange",
             linewidth=2, label="Validation loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], color="steelblue",
             linewidth=2, label="Train accuracy")
    ax2.plot(epochs, history["val_acc"], color="darkorange",
             linewidth=2, label="Validation accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to {save_path}")

    plt.show()


def set_seed(seed=42):
    """
    Sets random seeds for reproducibility across numpy, torch and CUDA.
    Call this at the start of any training script.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def print_class_distribution(dataset, split_name=""):
    """
    Prints the class distribution of a SoccerNetDataset instance.
    Useful for verifying label fractions and class balance.
    """
    from collections import Counter
    from dataset import IDX_TO_CLASS, BACKGROUND_IDX

    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)

    print(f"Class distribution{' — ' + split_name if split_name else ''}:")
    total_actions = 0
    for idx in sorted(counts.keys()):
        cls_name = IDX_TO_CLASS.get(idx, "Background")
        count = counts[idx]
        if idx != BACKGROUND_IDX:
            total_actions += count
        print(f"  {cls_name:<20} : {count:>6}")
    print(f"  {'Total actions':<20} : {total_actions:>6}")
    print(f"  {'Total samples':<20} : {len(labels):>6}")