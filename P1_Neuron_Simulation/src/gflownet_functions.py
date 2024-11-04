import torch
from torch import (
    exp,
    eye,
    nn, 
    Tensor, 
    tensor, 
    zeros_like
)


# def reward_function(original: Tensor, generated: Tensor) -> Tensor:
#     diff = nn.functional.mse_loss(generated, original)
#     return 1 / (diff + 1e-3)


# def reward_function(original: Tensor, generated: Tensor, matrix: Tensor) -> Tensor:
#     if (matrix.sum(dim=0) != 1).any() or (matrix.sum(dim=1) != 1).any():
#         return tensor(0.)
#     diff = nn.functional.mse_loss(generated, original)
#     return 1 / (diff + 1e-3)


# def reward_function(original: Tensor, generated: Tensor) -> Tensor:
#     if (generated.sum(dim=0) != 1).any() or (generated.sum(dim=1) != 1).any():
#         return tensor(0.)
#     return tensor(1.)

def reward_function(original: Tensor, generated: Tensor, matrix: Tensor) -> Tensor:
    # si la matriz tiene algún elemento en la diagonal, está mal
    if torch.diag(matrix).sum() > 0:
        return tensor(0., device=matrix.device)
    diff = nn.functional.mse_loss(generated, original)
    return 1 / (diff + 1e-3)


def column_reward(matrix: Tensor, std: float = 1.0) -> Tensor:
    n_neurons = len(matrix)
    n_neurons_connected = matrix.sum(dim=0)
    reward = exp(-(n_neurons_connected - 0.5 * n_neurons) / 2 * (std * std))
    return reward.sum()


def get_parents_flow_continuous(state: Tensor, step: int) -> Tensor:
    return state[step].sum()


def get_all_binary_matrices(n_spaces: int, flow_function: nn.Module) -> tuple[list[Tensor], list[Tensor]]:
    all_matrices = []
    all_flow = []
    for k in range(2**n_spaces):
        number = [int(i) for i in bin(k)[2:]]
        if (n := len(number)) < n_spaces:
            number = [0] * (n_spaces - n) + number
        matrix = tensor(number, dtype=torch.float, device=flow_function.device)
        all_matrices.append(matrix)
        all_flow.append(flow_function(matrix))
    return all_matrices, all_flow


def matrix_to_id(matrix: Tensor) -> int:
    powers = reversed(range(len(matrix)))
    powers = tensor(list(powers), dtype=float, device=matrix.device)
    return int((matrix * 2**powers).sum().item())


def get_parents_flow_binary(
        state: Tensor, 
        flow_function: nn.Module, 
        flows: list[Tensor] | None = None
    ):
    if flows is None:
        _, flows = get_all_binary_matrices(len(state), flow_function)
    
    if (state == zeros_like(state)).all():
        return tensor(0.)
    
    indexes = state.nonzero(as_tuple=True)[0]
    state_flow = tensor(0.)
    for i in indexes:
        parent = state.clone()
        parent[i] = 0
        parent_id = matrix_to_id(parent)
        flow = flows[parent_id][i] + get_parents_flow_binary(parent, flow_function, flows)
        state_flow = state_flow + flow
    return state_flow





# def get_parents_flow_binary(state: Tensor, flow_function: nn.Module) -> Tensor:
#     memo = dict()
#     def get_parents_flow_recursive(state: Tensor, memo: dict, flow_function: nn.Module) -> Tensor:
#         if (state == zeros_like(state)).all():
#             return tensor(0.)
#         indexes = state.nonzero(as_tuple=True)[0]
#         state_flow = tensor(0.)
#         for i in indexes:
#             parent = state.clone()
#             parent[i] = 0
#             if parent not in memo:
#                 memo[parent] = flow_function(parent)                        
#             flow = memo[parent][i] + get_parents_flow_recursive(parent, memo, flow_function)
#             state_flow = state_flow + flow
#         return state_flow
#     return get_parents_flow_recursive(state, memo, flow_function)
            
