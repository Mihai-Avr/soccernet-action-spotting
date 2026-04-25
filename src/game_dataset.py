import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from SoccerNet.utils import getListGames

from dataset import SELECTED_CLASSES, CLASS_TO_IDX, BACKGROUND_IDX

FEATURE_CONFIG = {
    "resnet": {
        "files": ("1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"),
        "fps": 2,
        "input_dim": 512
    },
    "baidu": {
        "files": ("1_baidu_soccer_embeddings.npy",
                  "2_baidu_soccer_embeddings.npy"),
        "fps": 1,
        "input_dim": 8576
    }
}


class SoccerNetGameDataset(Dataset):
    def __init__(self, data_path, split, feature_type="baidu",
                 label_radius=2, random_seed=42, max_games=None):
        """
        Loads full match halves for dense prediction training.
        Uses lazy loading — features are loaded from disk on demand
        to avoid RAM overflow with large feature sets like Baidu.

        data_path    : path to soccernet-data folder
        split        : train, valid or test
        feature_type : resnet or baidu
        label_radius : frames around annotation to mark as positive
        random_seed  : for reproducibility
        """
        assert feature_type in FEATURE_CONFIG, \
            f"feature_type must be one of {list(FEATURE_CONFIG.keys())}"

        self.data_path = data_path
        self.split = split
        self.feature_type = feature_type
        self.config = FEATURE_CONFIG[feature_type]
        self.fps = self.config["fps"]
        self.input_dim = self.config["input_dim"]
        self.label_radius = label_radius
        self.samples = []

        game_list = getListGames(split)
        if max_games is not None:
            import random
            random.seed(random_seed)
            game_list = random.sample(game_list, min(max_games, len(game_list)))
            print(f"  Using {len(game_list)} games (max_games={max_games})")

        print(f"Loading {split} split — {len(game_list)} games...")
        print(f"  Feature type : {feature_type}")
        print(f"  FPS          : {self.fps}")
        print(f"  Input dim    : {self.input_dim}")
        print(f"  Lazy loading : enabled (features loaded on demand)")

        total_action_frames = 0
        total_frames = 0

        for game in game_list:
            game_path = os.path.join(data_path, game)
            label_path = os.path.join(game_path, "Labels-v2.json")

            if not os.path.exists(label_path):
                continue

            with open(label_path, "r") as f:
                data = json.load(f)

            for half_idx, npy_file in enumerate(self.config["files"]):
                half = half_idx + 1
                npy_path = os.path.join(game_path, npy_file)

                if not os.path.exists(npy_path):
                    continue

                num_frames_approx = int(45 * 60 * self.fps)

                half_annotations = [
                    a for a in data["annotations"]
                    if int(a["gameTime"].split(" - ")[0]) == half
                    and a["label"] in SELECTED_CLASSES
                ]

                self.samples.append({
                    "npy_path": npy_path,
                    "annotations": half_annotations,
                    "game": game,
                    "half": half,
                    "label_radius": label_radius,
                    "fps": self.fps
                })

                total_action_frames += len(half_annotations) * (
                    2 * label_radius + 1
                )
                total_frames += num_frames_approx

        print(f"  Total halves: {len(self.samples)}")
        print(f"  Estimated action frames: ~{total_action_frames:,}")
        print(f"  Estimated total frames:  ~{total_frames:,}")
        print(f"  Estimated action ratio:  "
              f"~{100*total_action_frames/total_frames:.1f}%")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        features = np.load(sample["npy_path"]).astype(np.float32)
        num_frames = features.shape[0]

        labels = np.full(num_frames, BACKGROUND_IDX, dtype=np.int64)

        for ann in sample["annotations"]:
            time_str = ann["gameTime"].split(" - ")[1]
            minutes, seconds = map(int, time_str.split(":"))
            center_frame = int(
                (minutes * 60 + seconds) * sample["fps"]
            )
            center_frame = min(center_frame, num_frames - 1)
            cls_idx = CLASS_TO_IDX[ann["label"]]

            start = max(0, center_frame - sample["label_radius"])
            end = min(
                num_frames,
                center_frame + sample["label_radius"] + 1
            )
            labels[start:end] = cls_idx

        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return features_tensor, labels_tensor


def get_game_dataloader(dataset, shuffle=False, num_workers=2):
    """
    DataLoader for game-based dataset.
    Batch size is always 1 — one match half per batch.
    """
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )