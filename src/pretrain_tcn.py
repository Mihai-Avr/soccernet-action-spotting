import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from model import SoccerNetTCN
from game_dataset import SoccerNetGameDataset, get_game_dataloader
from utils import get_device, set_seed, save_checkpoint


def mask_features_tube_tcn(features, mask_ratio=0.75, tube_length=8):
    """
    Tube masking for full match half sequences.
    Same principle as VideoMAE but applied to 5400-frame sequences.

    features   : tensor of shape (1, seq_len, input_dim)
    mask_ratio : fraction of frames to mask
    tube_length: frames per tube (default 8 = 4 seconds at 2fps)

    Returns masked_features, mask, original
    """
    seq_len = features.shape[1]
    original = features.clone()

    num_tubes_total = seq_len // tube_length
    num_tubes_to_mask = max(1, int(num_tubes_total * mask_ratio))

    tube_indices = torch.randperm(num_tubes_total)[:num_tubes_to_mask]

    mask = torch.zeros(seq_len, dtype=torch.bool)
    for tube_idx in tube_indices:
        start = tube_idx.item() * tube_length
        end = min(start + tube_length, seq_len)
        mask[start:end] = True

    masked_features = features.clone()
    masked_features[0, mask, :] = 0.0

    return masked_features, mask, original


class TCNReconstructionHead(nn.Module):
    def __init__(self, d_model=256, output_dim=512):
        """
        Projects TCN encoder output back to input feature dimension.
        Used only in Stage 1 pretraining.
        """
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, x):
        return self.projection(x)


def pretrain_tcn(model, dataloader, num_epochs=50,
                 learning_rate=1e-3, mask_ratio=0.75,
                 tube_length=8, checkpoint_dir="checkpoints",
                 device=None):
    """
    Stage 1 pretraining for SoccerNetTCN using tube masking MFM.
    Processes full match halves instead of short windows.
    """
    if device is None:
        device = get_device()

    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)

    reconstruction_head = TCNReconstructionHead(
        d_model=256, output_dim=512
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) +
        list(reconstruction_head.parameters()),
        lr=learning_rate,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5
    )

    criterion = nn.MSELoss()
    best_loss = float("inf")
    history = []

    start_epoch = 1
    latest_path = os.path.join(checkpoint_dir, "pretrain_tcn_latest.pt")

    if os.path.exists(latest_path):
        print("Found existing checkpoint — checking if resume needed...")
        response = input("Resume from latest checkpoint? (y/n): ")
        if response.lower() == "y":
            ckpt = torch.load(
                latest_path, map_location=device, weights_only=False
            )
            model.load_state_dict(ckpt["model_state_dict"])
            reconstruction_head.load_state_dict(
                ckpt["reconstruction_head_state_dict"]
            )
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_loss = ckpt["loss"]
            history = ckpt.get("history", [])
            print(f"Resumed from epoch {ckpt['epoch']} "
                  f"(loss: {ckpt['loss']:.4f})")

    print(f"\nStarting TCN Stage 1 pretraining on {device}")
    print(f"  Epochs       : {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Mask ratio   : {mask_ratio}")
    print(f"  Tube length  : {tube_length} frames "
          f"({tube_length/2:.1f}s at 2fps)")
    print(f"  Halves/epoch : {len(dataloader)}")
    print("-" * 50)

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        reconstruction_head.train()
        total_loss = 0.0
        num_halves = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch:03d}", leave=False)

        for features, _ in progress:
            features = features.to(device)

            masked_features, mask, original = mask_features_tube_tcn(
                features, mask_ratio, tube_length
            )
            mask = mask.to(device)

            encoder_output = model.get_encoder_output(masked_features)
            reconstructed = reconstruction_head(encoder_output)

            masked_original = original[0, mask, :]
            masked_reconstructed = reconstructed[0, mask, :]
            loss = criterion(masked_reconstructed, masked_original)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) +
                list(reconstruction_head.parameters()),
                max_norm=1.0
            )
            optimizer.step()

            total_loss += loss.item()
            num_halves += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_halves
        scheduler.step()
        history.append(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:03d}/{num_epochs} | "
              f"Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "reconstruction_head_state_dict":
                reconstruction_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
            "history": history
        }, latest_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": avg_loss
            }, os.path.join(checkpoint_dir, "pretrain_tcn_best.pt"))
            print(f"  -> New best model saved (loss: {best_loss:.4f})")

    print("-" * 50)
    print(f"Pretraining complete. Best loss: {best_loss:.4f}")

    np.save(
        os.path.join(checkpoint_dir, "pretrain_tcn_history.npy"),
        np.array(history)
    )

    return model, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TCN Stage 1 Pretraining"
    )
    parser.add_argument("--data_path", type=str,
                        default="D:/soccernet-data")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="checkpoints/tcn")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--tube_length", type=int, default=8)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    print("Loading training dataset...")
    train_dataset = SoccerNetGameDataset(
        data_path=args.data_path,
        split="train",
        fps=2,
        label_radius=4
    )
    train_loader = get_game_dataloader(train_dataset, shuffle=True)

    model = SoccerNetTCN(
        input_dim=512,
        d_model=256,
        num_layers=8,
        kernel_size=3,
        dropout=0.1,
        num_classes=18
    )

    model, history = pretrain_tcn(
        model=model,
        dataloader=train_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        mask_ratio=args.mask_ratio,
        tube_length=args.tube_length,
        checkpoint_dir=args.checkpoint_dir,
        device=device
    )