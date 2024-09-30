import torch


def reward_function(original: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
    diff = torch.nn.functional.mse_loss(generated, original)
    # diff = (generated - original).square().sum()
    return 1 / (diff + 1)


def column_reward(matrix: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    n_neurons = len(matrix)
    n_neurons_connected = matrix.sum(dim=0)
    reward = torch.exp(-(n_neurons_connected - 0.5 * n_neurons) / 2 * (std * std))
    return reward.sum()


def get_parents_flow_continuous(state: torch.Tensor, step: int) -> torch.Tensor:
    return state[step].sum()


def get_parents_flow_binary(
    parents: list[torch.Tensor], flow_function: torch.nn.Module
) -> torch.Tensor:
    return sum(
        (flow_function(parent).sum() for parent in parents),
        start=torch.tensor(0.0, device=flow_function.device),
    )
