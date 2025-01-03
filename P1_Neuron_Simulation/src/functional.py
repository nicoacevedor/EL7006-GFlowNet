import numpy as np 
import torch


def sigmoid(x):
    
    return 1 / (1 + torch.exp(-x))


def correlation_for_all_neurons(X):
    """Computes the connectivity matrix for the all 
    neurons using correlations

    Args:
        X: the matrix of activities

    Returns:
        estimated_connectivity : estimated connectivity 
        for the selected neuron, of shape (n_neurons,)
    """

    n_neurons = len(X)

    S = np.concatenate([X[:, 1:], X[:, :-1]], axis=0)

    R = np.corrcoef(S)[:n_neurons, n_neurons:]

    return R


def LSE(x: torch.Tensor) -> torch.Tensor:
    ...


def gaussian_kernel(x: torch.Tensor, kernel_width: torch.Tensor | float) -> torch.Tensor:
    kernel_width = torch.tensor(kernel_width)
    arg = -(x).square() / (2 * kernel_width.square())
    return 0.39894 * arg.exp() / kernel_width


def correntropy(x: torch.Tensor, y: torch.Tensor, kernel_width: torch.Tensor | float) -> torch.Tensor:
    return gaussian_kernel(x - y, kernel_width).mean()
