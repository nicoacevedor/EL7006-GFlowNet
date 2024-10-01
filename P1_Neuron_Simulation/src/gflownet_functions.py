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
    return state[step].sum()


def get_all_binary_matrices(n_spaces: int, flow_function: torch.nn.Module) -> list[torch.Tensor]:
    all_matrices = []
    for k in range(2**n_spaces):
        number = [int(i) for i in bin(k)[2:]]
        if (n := len(number)) < n_spaces:
            number = [0] * (n_spaces - n) + number
        matrix = torch.tensor(number, dtype=torch.float, device=flow_function.device)
        all_matrices.append(flow_function(matrix))
    return all_matrices


def matrix_to_id(matrix: torch.Tensor) -> int:
    powers = reversed(range(len(matrix)))
    powers = torch.tensor(list(powers), dtype=torch.float, device=matrix.device)
    return int((matrix * 2**powers).sum().item())


def get_parents_flow_binary(
        state: torch.Tensor, 
        flow_function: torch.nn.Module, 
        matrices: list[torch.Tensor] | None = None
    ):
    if matrices is None:
        matrices = get_all_binary_matrices(len(state), flow_function)
    
    if (state == torch.zeros_like(state)).all():
        return torch.tensor(0.)
    
    indexes = state.nonzero(as_tuple=True)[0]
    state_flow = torch.tensor(0.)
    for i in indexes:
        parent = state.clone()
        parent[i] = 0
        parent_id = matrix_to_id(parent)
        flow = matrices[parent_id][i] + get_parents_flow_binary(parent, flow_function, matrices)
        state_flow = state_flow + flow
    return state_flow





# def get_parents_flow_binary(state: torch.Tensor, flow_function: torch.nn.Module) -> torch.Tensor:
#     memo = dict()
#     def get_parents_flow_recursive(state: torch.Tensor, memo: dict, flow_function: torch.nn.Module) -> torch.Tensor:
#         if (state == torch.zeros_like(state)).all():
#             return torch.tensor(0.)
#         indexes = state.nonzero(as_tuple=True)[0]
#         state_flow = torch.tensor(0.)
#         for i in indexes:
#             parent = state.clone()
#             parent[i] = 0
#             if parent not in memo:
#                 memo[parent] = flow_function(parent)                        
#             flow = memo[parent][i] + get_parents_flow_recursive(parent, memo, flow_function)
#             state_flow = state_flow + flow
#         return state_flow
#     return get_parents_flow_recursive(state, memo, flow_function)
            
