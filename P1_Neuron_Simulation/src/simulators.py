from src.functional import sigmoid
import numpy as np
import torch


class NetworkSystemSimulator(object):
    
    def __init__(self, random_state=7, prob=0.9, device: str = "cpu"):
        np.random.seed(random_state)
        self.p = prob     
        self.device = device

    def create_connectivity(self, n_neurons):
        """ Generate our NxN causal connectivity matrix.

        Args:
            n_neurons (int): the number of neurons in the system.
            random_state (int): random seed for reproducibility

        Returns:
            A (np.array): 0.1 sparse connectivity matrix
        """


        A_0 = np.random.choice(
            [0, 1], size=(n_neurons, n_neurons), p=[self.p, 1 - self.p]
        )

        _, eigen_values, _ = np.linalg.svd(A_0)

        if eigen_values[0] != 0 and not np.isnan(eigen_values[0]):
            return A_0 / (1.01 * eigen_values[0])

        return (1e-12) * np.ones_like(eigen_values)
    
    def simulate_neurons(self, A: torch.Tensor, timesteps: int = 5000, initial_value: torch.Tensor | None = None, perturb=False):
        """
        Simulates a dynamical system for the specified number 
        of neurons and timesteps,
        
        Args:
            A (np.array): the true connectivity matrix
            timesteps (int): the number of timesteps to
            simulate our system.
            perturb: Add a pattern of 1s and 0s each paired steps

        Returns:
            The results of the simulated system.
            - X has shape (n_neurons, timeteps)
        """
        n_neurons = len(A)
        if initial_value is None:
            X = torch.rand((n_neurons, timesteps), device=self.device)
        else:
            X = torch.zeros((n_neurons, timesteps), device=self.device)
            X[:, 0] = initial_value
        
        for t in range(timesteps - 1):
            
            if t % 2 == 0 and perturb:
                X[:, t] = torch.rand(n_neurons, device=self.device)
            
            epsilon = torch.randn(n_neurons, device=self.device)
            X[:, t + 1] = sigmoid(torch.matmul(A, X[:, t]) + epsilon) 

        return X