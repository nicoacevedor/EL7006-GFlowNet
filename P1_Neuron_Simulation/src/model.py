import torch.cuda as cuda
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import sigmoid


class FlowModel(nn.Module):
    def __init__(self, n_neurons: int, num_hid: int) -> None:
        super().__init__()
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.n_neurons = n_neurons
        # Red neuronal de 1 capa densa
        self.mlp = nn.Sequential(
            nn.Linear(n_neurons * n_neurons, num_hid),
            nn.LeakyReLU(),
            nn.Linear(num_hid, n_neurons * n_neurons),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.mlp(x)
        # return sigmoid(y)
        return y.sigmoid() * (1 - x)