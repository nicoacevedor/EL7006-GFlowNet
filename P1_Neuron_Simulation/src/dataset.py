from pathlib import Path

import torch
from torch.utils.data import Dataset


class NeuronDataset(Dataset):
    
    def __init__(self, filepath: str | Path, map_location: str = "cpu") -> None:
        path = Path(filepath)
        self.data = torch.load(path, map_location=map_location, weights_only=True)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]