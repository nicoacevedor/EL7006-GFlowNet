import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, Circle
from pathlib import Path

def pendulum_cart_ode(t, x, m, M, L, g, d, u_func):
    """
    Defines the ordinary differential equations for the pendulum-cart system.

    Args:
        t (float): Time
        x (array): State vector [cart position, cart velocity, pendulum angle, pendulum angular velocity]
        m (float): Mass of the pendulum
        M (float): Mass of the cart
        L (float): Length of the pendulum
        g (float): Gravitational acceleration
        d (float): Damping coefficient
        u_func (function): Control input function

    Returns:
        array: State derivatives [dx/dt, dv/dt, dθ/dt, dω/dt]
    """
    Sx = np.sin(x[2])
    Cx = np.cos(x[2])
    D = L * (M + m * (1 - Cx**2))  # Common denominator in [kg * m]
    
    dx = np.zeros(4)
    u = u_func(t)  # Control input at time t
    dx[0] = x[1]  # Change in cart position
    dx[1] = (1 / D) * (-m * L * g * Cx * Sx + L * (m * L * x[3]**2 * Sx - d * x[1]) + L * u)  # Change in cart velocity
    dx[2] = x[3]  # Change in pendulum angle
    dx[3] = (1 / D) * ((m + M) * g * Sx - Cx * (m * L * x[3]**2 * Sx - d * x[1]) - Cx * u)  # Change in pendulum angular velocity

    return dx

def draw_pendulum(ax, y, m, M, L, force=None):
    """
    Draws the pendulum-cart system on the given axis.

    Args:
        ax (matplotlib.axes.Axes): The axis to draw on
        y (array): Current state [cart position, cart velocity, pendulum angle, pendulum angular velocity]
        m (float): Mass of the pendulum
        M (float): Mass of the cart
        L (float): Length of the pendulum
        force (float, optional): Horizontal force on the cart
    """
    x = y[0]
    th = y[2]

    # Dimensions
    W = 1 * np.sqrt(M / 5)  # Cart width
    H = 0.5 * np.sqrt(M / 5)  # Cart height
    wr = 0.2  # Wheel radius
    mr = 0.3 * np.sqrt(m)  # Pendulum mass radius

    # Positions
    y_cart = wr / 2 + H / 2  # Vertical position of the cart
    w1x = x - 0.9 * W / 2
    w1y = 0
    w2x = x + 0.9 * W / 2 - wr
    w2y = 0

    px = x + L * np.sin(th)  # Pendulum x position
    py = y_cart - L * np.cos(th)  # Pendulum y position

    # Draw horizontal force arrow if provided
    if force is not None:
        F_x = force
        ax.annotate('', xy=(x + F_x, y_cart), xytext=(x, y_cart),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8, linestyle='dotted'))

    # Draw ground
    ax.plot([-10, 10], [0, 0], 'k', linewidth=2)

    # Draw cart
    cart = Rectangle((x - W / 2, y_cart - H / 2), W, H, facecolor=[0.5, 0.5, 1], linewidth=1.5)
    ax.add_patch(cart)

    # Draw wheels
    wheel1 = Circle((w1x+W/8, w1y), wr / 2, facecolor='black', linewidth=1.5)
    wheel2 = Circle((w2x+W/16, w2y), wr / 2, facecolor='black', linewidth=1.5)
    ax.add_patch(wheel1)
    ax.add_patch(wheel2)

    # Draw pendulum rod
    ax.plot([x, px], [y_cart, py], 'k', linewidth=2)

    # Draw pendulum mass
    pendulum_mass = Circle((px, py), mr / 2, facecolor=[1, 0.1, 0.1], linewidth=1.5)
    ax.add_patch(pendulum_mass)

    # Adjust limits and scale
    ax.set_xlim([-5, 5])
    ax.set_ylim([-2, 2.5])
    ax.set_aspect('equal', 'box')
    ax.axis('off')

def plot_simulation(m, M, L, g, d, sol, dt, path):
    """
    Creates an animation of the pendulum-cart system and saves it as a GIF.

    Args:
        m (float): Mass of the pendulum
        M (float): Mass of the cart
        L (float): Length of the pendulum
        g (float): Gravitational acceleration
        d (float): Damping coefficient
        sol (OdeResult): Solution of the ODE
        dt (float): Time step for the animation
        path (Path): Path to save the animation
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    u_expr = r'u(t) = sin(t)'  # Control function example
    title_str = (r'Pendulum Cart Simulation: ' 
                 f'$m = {m} [kg], M = {M} [kg], L = {L} [m], g = {g} [m/s^2], d = {d} [m]$ \n ${u_expr}$')
    
    def draw_pendulum_frame(i):
        ax.clear()
        u = u_func(sol.t[i])  # Control input at time t
        draw_pendulum(ax, sol.y[:, i], m, M, L, force=u)
        ax.set_title(title_str, fontsize=10, pad=20)

    anim = FuncAnimation(fig, draw_pendulum_frame, frames=len(sol.t), interval=dt * 1000, repeat=False)
    writer = PillowWriter(fps=10)
    anim.save(path / 'pendulum_cart.gif', writer=writer)
    plt.close()

def plot_all_states(states, m, M, L, path):
    """
    Plots all state variables of the pendulum-cart system over time.

    Args:
        states (array): Array of state variables over time
        m (float): Mass of the pendulum
        M (float): Mass of the cart
        L (float): Length of the pendulum
        path (Path): Path to save the plot
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    labels = ["Cart position (m)", "Cart velocity (m/s)", "Pendulum angle (rad)", "Pendulum angular velocity (rad/s)"]

    for i in range(4):
        axs[i].plot(states[i, :], label="Nonlinear")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)

    axs[2].set_title("Nonlinear pendulum evolution")
    axs[3].set_title("Nonlinear pendulum evolution")
    plt.tight_layout()
    plt.savefig(path / "pendulum_states.png")

if __name__ == "__main__":
    # System parameters
    m = 1
    M = 5
    L = 2
    g = -9.81
    d = 0.1
    initial_state = [0, 0, np.pi / 4, 0]  # Initial state [position, velocity, angle, angular velocity]
    t_span = (0, 10)  # Time interval for simulation
    dt = 0.1  # Time step for simulation
    save_data = False  # Option to save simulation data

    # Control function (can be adjusted)
    def u_func(t):
        return np.sin(t)
    
    # Solve the differential equation
    sol = solve_ivp(pendulum_cart_ode, t_span, initial_state, args=(m, M, L, g, d, u_func), t_eval=np.arange(t_span[0], t_span[1], dt))
    print(sol.message, sol.success, sol.t.shape, sol.y.shape)
    
    # Save data optionally
    if save_data:
        np.save("pendulum_states_and_time.npy", [sol.y, sol.t])
    
    # Create the path
    path = Path(__file__).parent
     
    # Plot simulation
    plot_simulation(m, M, L, g, d, sol, dt, path)

    # Plot all states
    plot_all_states(sol.y, m, M, L, path)