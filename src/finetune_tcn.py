import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from model import SoccerNetTCN
from game_dataset import SoccerNetGameDataset, get_game_dataloader, FEATURE_CONFIG
from dataset import CLASS_TO_IDX, SELECTED_CLASSES, BACKGROUND_IDX
from utils import get_device, set_seed, load_checkpoint


def compute_class_weights_dense(dataset, num_classes, device):
    """
    Computes inverse frequency class weights from annotation metadata.
    Works with lazy loading — uses annotation counts instead of
    loading full label arrays.
    """
    class_counts = torch.zeros(num_classes)

    background_frames_per_half = int(45 * 60 * dataset.fps)
    total_action_frames = 0

    for sample in dataset.samples:
        for ann in sample["annotations"]:
            cls_idx = CLASS_TO_IDX.get(ann["label"])
            if cls_idx is not None:
                frames = 2 * dataset.label_radius + 1
                class_counts[cls_idx] += frames
                total_action_frames += frames

    total_halves = len(dataset.samples)
    estimated_background = (
        total_halves * background_frames_per_half - total_action_frames
    )
    class_counts[BACKGROUND_IDX] = max(1, estimated_background)

    class_weights = 1.0 / (torch.sqrt(class_counts) + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes

    return class_weights.to(device)


def finetune_tcn_one_epoch(model, dataloader, optimizer,
                            criterion, device):
    """
    One epoch of dense prediction fine-tuning.
    Processes full match halves and computes per-frame loss.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_frames = 0

    progress = tqdm(dataloader, desc="Fine-tuning", leave=False)

    for features, labels in progress:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)

        logits_flat = logits.view(-1, logits.shape[-1])
        labels_flat = labels.view(-1)

        loss = criterion(logits_flat, labels_flat)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        optimizer.step()

        predictions = logits_flat.argmax(dim=1)
        total_correct += (predictions == labels_flat).sum().item()
        total_frames += labels_flat.shape[0]
        total_loss += loss.item()

        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_frames * 100
    return avg_loss, accuracy


def evaluate_tcn(model, dataloader, criterion, device):
    """
    Evaluates TCN on validation or test set.
    Returns average loss and frame-level accuracy.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_frames = 0

    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Evaluating",
                                      leave=False):
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            logits_flat = logits.view(-1, logits.shape[-1])
            labels_flat = labels.view(-1)

            loss = criterion(logits_flat, labels_flat)
            predictions = logits_flat.argmax(dim=1)

            total_correct += (predictions == labels_flat).sum().item()
            total_frames += labels_flat.shape[0]
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_frames * 100
    return avg_loss, accuracy


def finetune_tcn(model, train_dataset, valid_dataset,
                  num_epochs=50, learning_rate=1e-4,
                  checkpoint_dir="checkpoints/tcn",
                  patience=10, device=None, num_classes=18,
                  run_name="finetune_tcn",
                  resume_checkpoint=None,
                  early_stop_metric="val_loss",
                  use_reweighting=True,
                  use_label_smoothing=True):
    """
    Full Stage 2 dense prediction fine-tuning for SoccerNetTCN.
    """
    if device is None:
        device = get_device()

    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)

    smoothing = 0.1 if use_label_smoothing else 0.0

    if use_reweighting:
        print("Computing class weights...")
        class_weights = compute_class_weights_dense(
            train_dataset, num_classes, device
        )
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=smoothing
        )
        print(f"  Class reweighting : enabled")
        print(f"  Label smoothing   : {smoothing}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        print(f"  Class reweighting : disabled")
        print(f"  Label smoothing   : {smoothing}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_without_improvement = 0
    start_epoch = 1
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming from: {resume_checkpoint}")
        ckpt = torch.load(
            resume_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        best_val_acc = ckpt.get("val_acc", 0.0)
        history = ckpt.get("history", {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": []
        })
        print(f"  Resumed from epoch {ckpt['epoch']} "
            f"(val_acc: {ckpt.get('val_acc', 0):.1f}%)")

    train_loader = get_game_dataloader(train_dataset, shuffle=True)
    valid_loader = get_game_dataloader(valid_dataset, shuffle=False)


    print(f"\nStarting TCN Stage 2 fine-tuning on {device}")
    print(f"  Epochs        : {num_epochs} (patience={patience})")
    print(f"  Learning rate : {learning_rate}")
    print(f"  Train halves  : {len(train_dataset)}")
    print(f"  Valid halves  : {len(valid_dataset)}")
    print("-" * 60)

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss, train_acc = finetune_tcn_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate_tcn(
            model, valid_loader, criterion, device
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.1f}% | "
              f"LR: {current_lr:.6f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "history": history
        }, os.path.join(checkpoint_dir, f"{run_name}_latest.pt"))

        if early_stop_metric == "val_loss":
            improved = val_loss < best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
        else:
            improved = val_acc > best_val_acc
            best_val_acc = max(best_val_acc, val_acc)

        if improved:
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "history": history
            }, os.path.join(checkpoint_dir, f"{run_name}_best.pt"))
            print(f"  -> New best model saved "
                  f"(val_loss: {val_loss:.4f}, "
                  f"val_acc: {val_acc:.1f}%)")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement for "
                  f"{epochs_without_improvement}/{patience} epochs")

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    print("-" * 60)
    print(f"Fine-tuning complete.")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best val acc:  {best_val_acc:.1f}%")

    np.save(
        os.path.join(checkpoint_dir, f"{run_name}_history.npy"),
        history
    )

    return model, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TCN Stage 2 Fine-tuning"
    )
    parser.add_argument("--data_path", type=str,
                        default="D:/soccernet-data")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="checkpoints/tcn")
    parser.add_argument("--pretrain_checkpoint", type=str,
                        default=None)
    parser.add_argument("--run_name", type=str,
                        default="finetune_tcn")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--label_radius", type=int, default=2)
    parser.add_argument("--feature_type", type=str, default="baidu", choices=["resnet", "baidu"])
    parser.add_argument("--max_games", type=int, default=None)
    parser.add_argument("--valid_data_path", type=str, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--early_stop_metric", type=str,
                        default="val_loss",
                        choices=["val_loss", "val_acc"])
    parser.add_argument("--no_reweighting", action="store_true",
                        help="Disable class reweighting in loss function")
    parser.add_argument("--no_label_smoothing", action="store_true",
                        help="Disable label smoothing in loss function")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    print("Loading datasets...")
    train_dataset = SoccerNetGameDataset(
        data_path=args.data_path,
        split="train",
        feature_type=args.feature_type,
        label_radius=args.label_radius,
        max_games=args.max_games
    )
    valid_data_path = args.valid_data_path or args.data_path
    valid_dataset = SoccerNetGameDataset(
        data_path=valid_data_path,
        split="valid",
        feature_type=args.feature_type,
        label_radius=args.label_radius
    )
    
    input_dim = FEATURE_CONFIG[args.feature_type]["input_dim"]

    model = SoccerNetTCN(
        input_dim=input_dim,
        d_model=256,
        num_layers=8,
        kernel_size=3,
        dropout=0.1,
        num_classes=18
    )

    if args.pretrain_checkpoint:
        load_checkpoint(
            args.pretrain_checkpoint, model, device=device
        )
    else:
        print("No pretrained weights — training from scratch.")

    model, history = finetune_tcn(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        patience=args.patience,
        device=device,
        num_classes=18,
        run_name=args.run_name,
        resume_checkpoint=args.resume_checkpoint,
        early_stop_metric=args.early_stop_metric,
        use_reweighting=not args.no_reweighting,
        use_label_smoothing=not args.no_label_smoothing
    )