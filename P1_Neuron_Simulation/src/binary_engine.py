from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .simulators import NetworkSystemSimulator
from .gflownet_functions import reward_function, get_parents_flow_binary


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
                batch_loss = torch.tensor(0., device=self.device)
                for simulation in batch:
                    _, loss = self.sample_matrix(simulation)
                    batch_loss = batch_loss + loss
                batch_loss = batch_loss / len(batch)
                loss_item = batch_loss.detach().cpu().item()
                self.train_loss.append(loss_item)
                batch_loss.backward()
                opt.step()
                opt.zero_grad()
                pbar.set_postfix_str(f"loss: {loss_item:.3f}")
        self.save_training("training/binary/exp3")

    def sample_matrix(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_neurons = x.shape[0]
        matrix_length = n_neurons * n_neurons
        state = torch.zeros(matrix_length, device=self.device)
        flow_mismatch = torch.tensor(0., device=self.device)
        for t in range(matrix_length):
            flow_prediction = self.model(state)
            policy = flow_prediction / flow_prediction.sum()
            action = Categorical(probs=policy).sample()
            new_state = state.clone()
            new_state[t] = action
            parents_flow = get_parents_flow_binary(new_state, t+1, self.model)
            if t == matrix_length - 1:
                x_hat = self.neuron_simulator.simulate_neurons(
                    A=new_state.reshape(n_neurons, n_neurons),
                    timesteps=x.shape[1],
                    initial_value=x[:, 0]
                )
                reward = reward_function(x, x_hat)
                state_flow = torch.tensor(0., device=self.device)
                self.train_reward.append(reward.item())
            else:
                reward = torch.tensor(0., device=self.device)
                state_flow = self.model(new_state).sum()
            flow_mismatch = flow_mismatch + (parents_flow - state_flow - reward).square()
            state = new_state
        return state, flow_mismatch
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.sample_matrix(x)[0]
    
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
        
