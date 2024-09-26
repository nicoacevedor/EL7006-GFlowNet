import torch


def reward_function(original: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
    diff = torch.nn.functional.mse_loss(generated, original)
    return 1 / (diff + 1)


def column_reward(matrix: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    n_neurons = len(matrix)
    n_neurons_connected = matrix.sum(dim=0)
    reward = torch.exp(-(n_neurons_connected - 0.5 * n_neurons) / 2 * (std * std))
    return reward.sum()


def get_parents_flow_continuous(state: torch.Tensor, step: int) -> torch.Tensor:
    flow_list = []
    new_state = torch.zeros_like(state)
    for i in range(step):
        flow = state[i]
        flow_list.append(flow)
        new_state[i] = flow
    return torch.tensor(flow_list)


def get_parents_flow_binary(
    state: torch.Tensor, 
    step: int, 
    flow_function: torch.nn.Module
) -> torch.Tensor:
    flow_list = []
    new_state = torch.zeros_like(state)
    for i in range(step):
        index = state[i]
        flow = flow_function(new_state)
        flow_list.append(flow[index.int()])
        new_state = new_state.clone()
        new_state[i] = index
    return sum(flow_list, start=torch.tensor(0., device=flow_function.device))