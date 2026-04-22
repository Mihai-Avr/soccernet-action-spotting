import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from SoccerNet.utils import getListGames

from dataset import SELECTED_CLASSES, CLASS_TO_IDX, BACKGROUND_IDX


class SoccerNetGameDataset(Dataset):
    def __init__(self, data_path, split, fps=2,
                 label_radius=4, random_seed=42):
        """
        Loads full match halves for dense prediction training.
        Each sample is one match half — full 5400-frame sequence
        with per-frame labels.

        data_path    : path to soccernet-data folder
        split        : train, valid or test
        fps          : frames per second of features (default 2)
        label_radius : frames around annotation to mark as positive
                       (default 4 = ±2 seconds at 2fps)
        random_seed  : for reproducibility
        """
        self.data_path = data_path
        self.split = split
        self.fps = fps
        self.label_radius = label_radius
        self.samples = []

        game_list = getListGames(split)
        print(f"Loading {split} split — {len(game_list)} games...")

        for game in game_list:
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
                num_frames = features.shape[0]

                labels = np.full(num_frames, BACKGROUND_IDX, dtype=np.int64)

                half_annotations = [
                    a for a in data["annotations"]
                    if int(a["gameTime"].split(" - ")[0]) == half
                    and a["label"] in SELECTED_CLASSES
                ]

                for ann in half_annotations:
                    time_str = ann["gameTime"].split(" - ")[1]
                    minutes, seconds = map(int, time_str.split(":"))
                    center_frame = int((minutes * 60 + seconds) * fps)
                    center_frame = min(center_frame, num_frames - 1)

                    cls_idx = CLASS_TO_IDX[ann["label"]]

                    start = max(0, center_frame - label_radius)
                    end = min(num_frames, center_frame + label_radius + 1)
                    labels[start:end] = cls_idx

                self.samples.append({
                    "features": features,
                    "labels": labels,
                    "game": game,
                    "half": half,
                    "num_frames": num_frames
                })

        print(f"  Total halves loaded: {len(self.samples)}")

        total_action_frames = sum(
            (s["labels"] != BACKGROUND_IDX).sum()
            for s in self.samples
        )
        total_frames = sum(s["num_frames"] for s in self.samples)
        print(f"  Total frames: {total_frames:,}")
        print(f"  Action frames: {total_action_frames:,} "
              f"({100*total_action_frames/total_frames:.1f}%)")
        print(f"  Background frames: "
              f"{total_frames - total_action_frames:,} "
              f"({100*(total_frames-total_action_frames)/total_frames:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = torch.tensor(
            sample["features"], dtype=torch.float32
        )
        labels = torch.tensor(
            sample["labels"], dtype=torch.long
        )
        return features, labels


def get_game_dataloader(dataset, shuffle=False):
    """
    DataLoader for game-based dataset.
    Batch size is always 1 — one match half per batch.
    """
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=0
    )