from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .simulators import NetworkSystemSimulator
from .gflownet_functions import (
    correntropy_reward,
    get_parents_flow_binary,
    get_all_binary_matrices
)


class GFlowBinaryEngine:
    
    def __init__(self, model: torch.nn.Module, neuron_simulator: NetworkSystemSimulator, device: str | None = None) -> None:
        self.model = model
        self.neuron_simulator = neuron_simulator
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)
        self.train_loss = []
        self.train_reward = []

    def fit(
            self, 
            train_dataset: Dataset,
            n_epochs: int = 1,
            batch_size: int = 1,
            lr: float = 0.001,
            output_path :str = ""
        ) -> None:
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)  # type: ignore
        self.model.train()
        sample_matrix_batch = torch.vmap(self.sample_matrix, randomness="different")
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
        )
        print(f"Using device {self.device}")
        for epoch in tqdm(range(n_epochs), desc="Training..."):
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch:<5}", leave=False)
            for batch in pbar:
                presampled_flows, _ = get_all_binary_matrices(9, self.model)
                batch_loss = torch.tensor(0., device=self.device)
                for i, simulation in enumerate(batch):
                    _, loss = self.sample_matrix(simulation, presampled_flows)
                    batch_loss = batch_loss + loss
                    pbar.set_postfix(dict(loss=loss.detach().cpu().item(), simulation=i))
                batch_loss = batch_loss / len(batch)
                loss_item = batch_loss.detach().cpu().item()
                self.train_loss.append(loss_item)
                batch_loss.backward()
                opt.step()
                opt.zero_grad()
        self.save_training(output_path)

    def apply_action(self, state: Tensor, policy: Tensor) -> Tensor:
        change = Categorical(probs=policy).sample()
        new_state = state.clone()
        new_state[change] = 1
        return new_state
    
    @staticmethod
    def reward_function(x: Tensor, y: Tensor, **kwargs) -> Tensor:
        return correntropy_reward(x, y, kwargs["kernel_width"], kwargs["matrix"])

    def sample_matrix(self, x: Tensor, presampled_flows: list[Tensor] | None) -> tuple[Tensor, Tensor]:
        n_neurons = x.shape[0]
        matrix_length = n_neurons * n_neurons
        state = torch.zeros(matrix_length, device=self.device)
        flow_mismatch = torch.tensor(0., device=self.device)
        for t in range(matrix_length):
            with torch.no_grad():
                flow_prediction = self.model(state)
            policy = flow_prediction / flow_prediction.sum()
            new_state = self.apply_action(state, policy)
            parents_flow = get_parents_flow_binary(new_state, self.model, presampled_flows)
            if (t == matrix_length - 1) or (new_state == state).all():
                matrix = new_state.reshape(n_neurons, n_neurons)
                x_hat = self.neuron_simulator.simulate_neurons(
                    A=matrix,
                    timesteps=x.shape[1],
                    initial_value=x[:, 0]
                )
                reward = self.reward_function(x, x_hat, kernel_width=0.1, matrix=matrix)
                state_flow = torch.tensor(0., device=self.device)
                self.train_reward.append(reward.item())
                break
            else:
                reward = torch.tensor(0., device=self.device)
                state_flow = self.model(new_state).sum()
            flow_mismatch = flow_mismatch + (parents_flow - state_flow - reward).square()
            state = new_state
        return state, flow_mismatch
        
    def __call__(self, x: Tensor, presampled_flows: list[Tensor] | None = None) -> Tensor:
        return self.sample_matrix(x, presampled_flows)[0]
    
    def save_metrics_plot(self, path: Path) -> None:
        matplotlib.use("Agg")
        plt.figure(figsize=(14, 8))
        plt.plot(self.train_loss)
        plt.savefig(path/"loss.png")
        plt.figure(figsize=(14, 8))
        plt.plot(self.train_reward)
        plt.savefig(path/"reward.png")

    def save_n_samples(self, n_samples: int, dim: tuple[int, int], path: Path) -> None:
        f, ax = plt.subplots(dim[0], dim[1], figsize=(8, 8))
        with torch.no_grad():
            sampled_matrices = (
                self.__call__(torch.empty(3, 100, device=self.device)) for _ in range(n_samples)
            )
        for i, matrix in enumerate(sampled_matrices):
            plt.sca(ax[i // dim[1], i % dim[1]])
            plt.imshow(matrix.reshape(3, 3).cpu(), cmap="coolwarm")
            plt.axis("off")
        plt.savefig(path/"samples.png")

    def save_reward_histogram(self, path: Path) -> None:
        matplotlib.use("Agg")
        plt.figure(figsize=(14, 8))
        plt.hist(self.train_reward)
        plt.savefig(path/"histogram.png")

    def save_training(self, folder_path: str) -> None:
        path = Path(folder_path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path/"model.pt")
        self.save_metrics_plot(path)
        self.save_n_samples(64, (8, 8), path)
        self.save_reward_histogram(path)
