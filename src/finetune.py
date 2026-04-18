import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import SoccerNetDataset, get_dataloader
from model import SoccerNetTransformer
from utils import get_device, load_checkpoint, set_seed, plot_training_curves

def compute_class_weights(dataset, num_classes, device):
    """
    Computes inverse frequency class weights for weighted cross entropy.
    
    dataset    : SoccerNetDataset instance
    num_classes: total number of classes including background
    device     : torch device
    """
    labels = [label for _, label in dataset.samples]
    class_counts = torch.zeros(num_classes)
    
    for label in labels:
        class_counts[label] += 1
    
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    return class_weights.to(device)


def finetune_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Runs one full epoch of Stage 2 fine-tuning.
    Returns average training loss and accuracy.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Fine-tuning", leave=False)

    for batch_windows, batch_labels in progress_bar:
        batch_windows = batch_windows.to(device)
        batch_labels = batch_labels.to(device)

        logits = model(batch_windows)
        loss = criterion(logits, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == batch_labels).sum().item()
        total += batch_labels.size(0)

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on a validation or test set.
    Returns average loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_windows, batch_labels in dataloader:
            batch_windows = batch_windows.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_windows)
            loss = criterion(logits, batch_labels)

            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100
    return avg_loss, accuracy




def finetune(model, train_dataset, valid_dataset, num_epochs=30,
             learning_rate=1e-4, batch_size=32, checkpoint_dir="checkpoints",
             patience=7, device=None, num_classes=18, run_name="finetune", resume_checkpoint=None):
    """
    Full Stage 2 fine-tuning loop with early stopping and checkpointing.

    model          : SoccerNetTransformer with pretrained encoder weights
    train_dataset  : SoccerNetDataset for training split
    valid_dataset  : SoccerNetDataset for validation split
    num_epochs     : maximum number of fine-tuning epochs
    learning_rate  : learning rate (lower than Stage 1)
    batch_size     : samples per batch
    checkpoint_dir : directory to save checkpoints
    patience       : early stopping patience in epochs
    device         : torch device
    num_classes    : total number of classes including background
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)

    class_weights = compute_class_weights(train_dataset, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    train_loader = get_dataloader(
        train_dataset,
        batch_size=batch_size,
        use_weighted_sampler=True
    )
    valid_loader = get_dataloader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    start_epoch = 1
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        ckpt = load_checkpoint(resume_checkpoint, model, optimizer, device=device)
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_loss"]
        history = ckpt["history"]

    print(f"Starting Stage 2 fine-tuning on {device}")
    print(f"  Epochs        : {num_epochs} (patience={patience})")
    print(f"  Learning rate : {learning_rate}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Train samples : {len(train_dataset)}")
    print(f"  Valid samples : {len(valid_dataset)}")
    print("-" * 60)

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss, train_acc = finetune_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        val_loss, val_acc = evaluate(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            device=device
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | "
              f"LR: {current_lr:.6f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "history": history
        }, os.path.join(checkpoint_dir, f"{run_name}_latest.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc
            }, os.path.join(checkpoint_dir, f"{run_name}_best.pt"))
            print(f"  -> New best model saved "
                  f"(val_loss: {best_val_loss:.4f}, val_acc: {val_acc:.1f}%)")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement for "
                  f"{epochs_without_improvement}/{patience} epochs")

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    print("-" * 60)
    print(f"Fine-tuning complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return model, history


if __name__ == "__main__":
    import argparse
    from dataset import SoccerNetDataset, get_dataloader
    from model import SoccerNetTransformer

    parser = argparse.ArgumentParser(description="Stage 2 Fine-tuning")
    parser.add_argument("--data_path", type=str, default="D:/soccernet-data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="finetune")
    parser.add_argument("--pretrain_checkpoint", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--label_fraction", type=float, default=1.0)
    parser.add_argument("--num_classes", type=int, default=18)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    print("Loading datasets...")
    train_dataset = SoccerNetDataset(
        data_path=args.data_path,
        split="train",
        window_size=args.window_size,
        overlap=0,
        label_fraction=args.label_fraction
    )
    valid_dataset = SoccerNetDataset(
        data_path=args.data_path,
        split="valid",
        window_size=args.window_size,
        overlap=0
    )

    model = SoccerNetTransformer(
        input_dim=512,
        d_model=384,
        num_heads=4,
        num_layers=3,
        dim_feedforward=768,
        dropout=0.1,
        num_classes=18
    )

    if args.pretrain_checkpoint:
        load_checkpoint(args.pretrain_checkpoint, model, device=device)
    else:
        print("No pretrained weights loaded — training from scratch.")

    model, history = finetune(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        patience=args.patience,
        device=device,
        num_classes=18,
        run_name=args.run_name,
        resume_checkpoint=args.resume_checkpoint
    )

    print("\nSaving fine-tuning history...")
    history_path = os.path.join(args.checkpoint_dir, f"{args.run_name}_history.npy")
    np.save(history_path, history)
    plot_training_curves(
        history,
        title=f"Training curves — {args.run_name}",
        save_path=f"results/figures/curves_{args.run_name}.png"
    )
    print("Done.")