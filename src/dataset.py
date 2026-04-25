import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from SoccerNet.utils import getListGames
from collections import Counter

SELECTED_CLASSES = [
    "Ball out of play",
    "Throw-in",
    "Foul",
    "Indirect free-kick",
    "Clearance",
    "Shots on target",
    "Shots off target",
    "Corner",
    "Substitution",
    "Kick-off",
    "Direct free-kick",
    "Offside",
    "Yellow card",
    "Goal",
    "Penalty",
    "Red card",
    "Yellow->red card"
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(SELECTED_CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}
BACKGROUND_IDX = len(SELECTED_CLASSES)


class SoccerNetDataset(Dataset):
    def __init__(self, data_path, split, window_size=60, overlap=0.5,
                 label_fraction=1.0, random_seed=42):
        """
        data_path   : path to soccernet-data folder on the HDD
        split       : "train", "valid", or "test"
        window_size : number of frames per window (60 = 30 seconds at 2fps)
        overlap     : how much consecutive windows overlap (0.5 = 50%)
        """
        self.data_path = data_path
        self.split = split
        self.window_size = window_size
        self.overlap = overlap
        self.samples = []
        self.label_fraction = label_fraction
        self.random_seed = random_seed

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
                npy_path = os.path.join(game_path, f"{half}_ResNET_TF2_PCA512.npy")
                if not os.path.exists(npy_path):
                    continue

                features = np.load(npy_path).astype(np.float32)
                num_frames = features.shape[0]

                half_annotations = [
                    a for a in data["annotations"]
                    if int(a["gameTime"].split(" - ")[0]) == half
                    and a["label"] in SELECTED_CLASSES
                ]

                action_frames = {}
                for ann in half_annotations:
                    time_str = ann["gameTime"].split(" - ")[1]
                    minutes, seconds = map(int, time_str.split(":"))
                    frame_idx = int((minutes * 60 + seconds) * 2)
                    frame_idx = min(frame_idx, num_frames - 1)
                    action_frames[frame_idx] = CLASS_TO_IDX[ann["label"]]

                action_frame_set = set(action_frames.keys())

                action_samples = []
                for ann_frame, class_idx in action_frames.items():
                    start = max(0, ann_frame - window_size // 2)
                    end = start + window_size
                    if end > num_frames:
                        end = num_frames
                        start = end - window_size
                    window = features[start:end]
                    action_samples.append((window, class_idx))

                if label_fraction < 1.0:
                    rng = np.random.RandomState(random_seed)
                    n_keep = max(1, int(len(action_samples) * label_fraction))
                    indices = rng.choice(
                        len(action_samples), size=n_keep, replace=False
                    )
                    action_samples = [action_samples[i] for i in indices]

                self.samples.extend(action_samples)

                step = max(1, int(window_size * (1 - overlap)))
                for start in range(0, num_frames - window_size, step):
                    end = start + window_size
                    center = (start + end) // 2
                    too_close = any(
                        abs(center - af) < window_size // 2
                        for af in action_frame_set
                    )
                    if not too_close:
                        window = features[start:end]
                        self.samples.append((window, BACKGROUND_IDX))

        print(f"  Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window, label = self.samples[idx]
        return torch.tensor(window, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    

def get_dataloader(dataset, batch_size=32, shuffle=False, use_weighted_sampler=False):

    if use_weighted_sampler:
        labels = [label for _, label in dataset.samples]
        class_counts = Counter(labels)

        class_weights = {
            cls_idx: 1.0 / count
            for cls_idx, count in class_counts.items()
        }

        sample_weights = [class_weights[label] for label in labels]
        sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )