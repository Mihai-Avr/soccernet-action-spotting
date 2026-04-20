import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils import get_device, set_seed

def mask_features(batch, mask_ratio=0.15):
    """
    Randomly masks a fraction of frames in each window.
    
    batch      : tensor of shape (batch_size, window_size, input_dim)
    mask_ratio : fraction of frames to mask (default 0.15 = 15%)
    
    Returns:
        masked_batch  : batch with masked positions set to zero
        mask          : boolean tensor, True where frames were masked
        original      : original unmasked batch (reconstruction targets)
    """
    batch_size, window_size, input_dim = batch.shape
    
    original = batch.clone()
    masked_batch = batch.clone()
    
    mask = torch.zeros(batch_size, window_size, dtype=torch.bool)
    
    num_masked = max(1, int(window_size * mask_ratio))
    
    for i in range(batch_size):
        masked_indices = torch.randperm(window_size)[:num_masked]
        mask[i, masked_indices] = True
        masked_batch[i, masked_indices] = 0.0
    
    return masked_batch, mask, original

def mask_features_tube(batch, mask_ratio=0.75, tube_length=4):
    """
    Tube masking strategy inspired by VideoMAE.
    Masks contiguous temporal blocks (tubes) rather than
    individual random frames. Vectorized implementation.

    batch        : tensor of shape (batch_size, window_size, input_dim)
    mask_ratio   : fraction of frames to mask (default 0.75 = 75%)
    tube_length  : number of consecutive frames per tube (default 4 = 2s)

    Returns:
        masked_batch : batch with masked positions set to zero
        mask         : boolean tensor, True where frames were masked
        original     : original unmasked batch (reconstruction targets)
    """
    batch_size, window_size, input_dim = batch.shape

    original = batch.clone()
    mask = torch.zeros(batch_size, window_size, dtype=torch.bool)

    num_tubes_total = window_size // tube_length
    num_tubes_to_mask = max(1, int(num_tubes_total * mask_ratio))

    for i in range(batch_size):
        tube_indices = torch.randperm(num_tubes_total)[:num_tubes_to_mask]
        for tube_idx in tube_indices:
            start = tube_idx * tube_length
            end = min(start + tube_length, window_size)
            mask[i, start:end] = True

    masked_batch = batch.clone()
    masked_batch[mask] = 0.0

    return masked_batch, mask, original

class ReconstructionHead(nn.Module):
    def __init__(self, d_model=384, output_dim=512):
        """
        Projects encoder output back to original feature dimension.
        Used only in Stage 1, discarded before Stage 2.
        
        d_model    : Transformer encoder output dimension
        output_dim : original input feature dimension to reconstruct
        """
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, x):
        return self.projection(x)


def pretrain_one_epoch(model, reconstruction_head, dataloader,
                       optimizer, device, mask_ratio=0.75,
                       tube_length=4, masking_strategy="tube"): # mask_ratio = 0.75 for tube masking, 0.15 for random masking
    """
    Runs one full epoch of Stage 1 MFM pretraining.
    Returns the average reconstruction loss for the epoch.
    """
    model.train()
    reconstruction_head.train()

    criterion = nn.MSELoss()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Pretraining", leave=False)

    for batch_windows, _ in progress_bar:
        batch_windows = batch_windows.to(device)

        if masking_strategy == "tube":
            masked_batch, mask, original = mask_features_tube(
                batch_windows, mask_ratio, tube_length
            )
        else:
            masked_batch, mask, original = mask_features(
                batch_windows, mask_ratio
            )
            
        mask = mask.to(device)

        encoder_output = model.get_encoder_output(masked_batch)
        reconstructed = reconstruction_head(encoder_output)

        masked_original = original[mask]
        masked_reconstructed = reconstructed[mask]

        loss = criterion(masked_reconstructed, masked_original)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(reconstruction_head.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches



def pretrain(model, dataloader, num_epochs=10, learning_rate=1e-3,
             mask_ratio=0.75, tube_length=4,
             masking_strategy="tube",
             checkpoint_dir="checkpoints", device=None): # mask_ratio = 0.75 for tube masking, 0.15 for random masking
    """
    Full Stage 1 pretraining loop with checkpointing.
    
    model          : SoccerNetTransformer instance
    dataloader     : training DataLoader (no weighted sampler needed)
    num_epochs     : number of pretraining epochs
    learning_rate  : initial learning rate
    mask_ratio     : fraction of frames to mask
    tube_length    : length of tubes for tube masking
    masking_strategy : strategy for masking ("tube" or "random")
    checkpoint_dir : directory to save checkpoints
    device         : torch device (cuda or cpu)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(checkpoint_dir, exist_ok=True)

    model = model.to(device)

    reconstruction_head = ReconstructionHead(d_model=384, output_dim=512).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(reconstruction_head.parameters()),
        lr=learning_rate,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-5
    )

    best_loss = float("inf")
    start_epoch = 1
    
    if os.path.exists(os.path.join(checkpoint_dir, "pretrain_latest.pt")):
        print("Found existing checkpoint — checking if resume is needed...")
        response = input("Resume from latest checkpoint? (y/n): ")
        if response.lower() == "y":
            ckpt = torch.load(
                os.path.join(checkpoint_dir, "pretrain_latest.pt"),
                map_location=device,
                weights_only=False
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
            print(f"Resumed from epoch {ckpt['epoch']} "
                f"(loss: {ckpt['loss']:.4f})")

    history = []

    print(f"Starting Stage 1 pretraining on {device}")
    print(f"  Epochs           : {num_epochs}")
    print(f"  Learning rate    : {learning_rate}")
    print(f"  Masking strategy : {masking_strategy}")
    print(f"  Mask ratio       : {mask_ratio}")
    print(f"  Tube length      : {tube_length} frames ({tube_length/2:.1f}s)")
    print(f"  Batches/epoch    : {len(dataloader)}")
    print("-" * 50)

    for epoch in range(start_epoch, num_epochs + 1):
        avg_loss = pretrain_one_epoch(
            model=model,
            reconstruction_head=reconstruction_head,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            mask_ratio=mask_ratio,
            tube_length=tube_length,
            masking_strategy=masking_strategy
)

        scheduler.step()
        history.append(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:02d}/{num_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"LR: {current_lr:.6f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "reconstruction_head_state_dict": reconstruction_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss
        }, os.path.join(checkpoint_dir, "pretrain_latest.pt"))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": avg_loss
            }, os.path.join(checkpoint_dir, "pretrain_best.pt"))
            print(f"  -> New best model saved (loss: {best_loss:.4f})")

    print("-" * 50)
    print(f"Pretraining complete. Best loss: {best_loss:.4f}")

    return model, history

if __name__ == "__main__":
    import argparse
    from dataset import SoccerNetDataset, get_dataloader
    from model import SoccerNetTransformer

    parser = argparse.ArgumentParser(description="Stage 1 MFM Pretraining")
    parser.add_argument("--data_path", type=str, default="D:/soccernet-data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--masking_strategy", type=str, default="tube",
                    choices=["tube", "random"])
    parser.add_argument("--tube_length", type=int, default=4)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    print("Loading training dataset...")
    train_dataset = SoccerNetDataset(
        data_path=args.data_path,
        split="train",
        window_size=args.window_size,
        overlap=0
    )

    pretrain_loader = get_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        use_weighted_sampler=False
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

    model, history = pretrain(
        model=model,
        dataloader=pretrain_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        mask_ratio=args.mask_ratio,
        tube_length=args.tube_length,
        masking_strategy=args.masking_strategy,
        checkpoint_dir=args.checkpoint_dir,
        device=device
    )

    print("\nSaving loss history...")
    history_path = os.path.join(args.checkpoint_dir, "pretrain_history.npy")
    np.save(history_path, np.array(history))
    print(f"Loss history saved to {history_path}")