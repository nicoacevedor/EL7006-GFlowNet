from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
from torch import Tensor, tensor



def plot_neural_activity(X, max_timesteps=10):
    """Plot first 10 timesteps of neural activity

    Args:
        X (ndarray): neural activity (n_neurons by timesteps)

    """
    figure, axis = plt.subplots()
    
    im = axis.imshow(X[:, :max_timesteps])
    
    divider = make_axes_locatable(axis)
    
    cax1 = divider.append_axes("right", size="5%", pad=0.15)
    
    plt.colorbar(im, cax=cax1)
    
    axis.set(
        xlabel='Timestep', ylabel='Neuron', 
        title='Simulated Neural Activity'
    )
    
    return figure, axis
