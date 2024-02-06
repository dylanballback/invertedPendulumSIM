import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import numpy as np
from scipy.signal import place_poles
from scipy.integrate import cumtrapz
import time

"""
12/5/23 Breaking up each thing into its own fuction so I can try different controller types like a PID.

Broken Doesn't work :()
"""

# Physical parameters of the pendulum
m_p = 0.5  # Mass of the pendulum (kg)
l = 0.15   # Length to pendulum's center of mass (m)
I_p = m_p * l**2 / 3  # Moment of inertia of the pendulum (kg.m^2)
I_w = 0.001953125  # Moment of inertia of the reaction wheel (kg.m^2)
g = 9.81  # Gravitational acceleration (m/s^2)

# Linearized system matrices
A = np.array([[0, 1, 0],
              [m_p * g * l / (I_p + m_p * l**2), 0, 0],
              [0, 0, 0]])
B = np.array([[0],
              [-1 / (I_p + m_p * l**2)],
              [1 / I_w]])


def pid_controller(state, Kp, Ki, Kd, integral, last_error, dt):
    """
    PID controller function.

    Parameters:
    - state: The state vector of the system.
    - Kp, Ki, Kd: PID gains.
    - integral: Integral term (should be stored and updated externally).
    - last_error: Last error term (should be stored and updated externally).
    - dt: Time step duration.
    """
    error = state[0]  # Assuming the error is the deviation in theta
    integral += error * dt
    derivative = (error - last_error) / dt
    output = Kp * error + Ki * integral + Kd * derivative
    last_error = error
    return output, integral, last_error

def pid_wrapper(state):
    global integral, last_error
    torque, integral, last_error = pid_controller(state, Kp, Ki, Kd, integral, last_error, dt)
    return torque  # Ensure a single value is returned

# LQR Controller
def lqr_controller(state):
    """
    LQR controller function.

    Parameters:
    - state: The state vector of the system.
    """
    # Desired pole locations
    poles = np.array([-1, -1.1, -1.2])
    K = place_poles(A, B, poles).gain_matrix
    torque = -K.dot(state)
    return torque[0]  # Ensure a single value is returned

def simulate_inverted_pendulum(controller, initial_conditions, simulation_duration=10, time_steps=300):
    """
    Simulate an inverted pendulum with a given controller.

    Parameters:
    - controller: A function that takes the state vector and returns the control input (torque).
    - initial_conditions: Initial state of the system.
    - simulation_duration: Duration of the simulation in seconds.
    - time_steps: Number of time steps in the simulation.
    """
    # Define the physical parameters
    m_p = 0.5  # Mass of the pendulum (kg)
    l = 0.15  # Length to pendulum's center of mass (m)
    I_p = m_p * l**2 / 3  # Moment of inertia of the pendulum (kg.m^2)
    I_w = 0.001953125  # Moment of inertia of the reaction wheel (kg.m^2)
    g = 9.81  # Gravitational acceleration (m/s^2)

    # Equations of motion with the controller
    def equations_of_motion(t, y):
        theta, theta_dot, phi_dot = y
        torque = controller(y)
        d_theta_dot = (m_p * g * l * np.sin(theta) - torque) / (I_p + m_p * l**2)
        d_phi_dot = torque / I_w
        return [theta_dot, d_theta_dot, d_phi_dot]

    # Time span for the simulation
    t_span = [0, simulation_duration]
    t_eval = np.linspace(*t_span, time_steps)

    # Solve the differential equations
    solution = solve_ivp(equations_of_motion, t_span, initial_conditions, t_eval=t_eval)

    return solution


# Graphing Function

# Graphing and Animation Function
def setup_and_animate_graphs(solution, l, K):
    # Set up the figure with a specified figure size
    fig = plt.figure(figsize=(10, 12))

    # Create a grid layout with 4 rows and 2 columns
    gs = fig.add_gridspec(4, 2)

    # Assign the subplots to the grid
    ax1 = fig.add_subplot(gs[:, 0])  # Inverted pendulum plot spanning all rows
    ax2 = fig.add_subplot(gs[0, 1])  # Theta error plot
    ax3 = fig.add_subplot(gs[1, 1])  # Torque plot
    ax4 = fig.add_subplot(gs[2, 1])  # Angular velocity plot
    ax5 = fig.add_subplot(gs[3, 1])  # Angular acceleration plot

    # Set plot limits and labels
    ax1.set_xlim((-l, l))
    ax1.set_ylim((-l, 1.5 * l))
    ax1.set_aspect('equal')  # Ensure the wheel looks like a circle

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Theta Error (degrees)')

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Torque (Nm)')

    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Angular Velocity (turns/s)')

    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Angular Acceleration (turns/s²)')

    # Initializations for plots
    line, = ax1.plot([], [], 'k-', lw=2)
    wheel_radius = 0.05 * l
    wheel = plt.Circle((0, l), wheel_radius, color='blue', fill=False)
    ax1.add_patch(wheel)

    line_error, = ax2.plot([], [], 'r-', lw=2)
    line_torque, = ax3.plot([], [], 'b-', lw=2)
    line_angular_velocity, = ax4.plot([], [], 'g-', lw=2)
    line_angular_acceleration, = ax5.plot([], [], 'm-', lw=2)

    # Define init function for animation
    def init():
        line.set_data([], [])
        wheel.center = (0, l)
        line_error.set_data([], [])
        line_torque.set_data([], [])
        line_angular_velocity.set_data([], [])
        line_angular_acceleration.set_data([], [])
        return line, wheel, line_error, line_torque, line_angular_velocity, line_angular_acceleration

    # Define animate function
    def animate(i):
        x = [0, l * np.sin(solution.y[0, i])]
        y = [0, -l * np.cos(solution.y[0, i])]
        line.set_data(x, y)
        wheel.center = (x[1], y[1])

        line_error.set_data(solution.t[:i+1], np.degrees(solution.y[0, :i+1]))
        line_torque.set_data(solution.t[:i+1], -K @ solution.y[:, :i+1])
        line_angular_velocity.set_data(solution.t[:i+1], solution.y[2, :i+1] / (2 * np.pi))

        if i == 0:
            angular_acceleration = [0]
        else:
            angular_acceleration = np.gradient(solution.y[2, :i+1], solution.t[:i+1]) / (2 * np.pi)

        line_angular_acceleration.set_data(solution.t[:i+1], angular_acceleration)

        return line, wheel, line_error, line_torque, line_angular_velocity, line_angular_acceleration


    # Create the animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(solution.t), interval=20, blit=True)

    return anim, fig





# Graphing and Animation Function
def setup_and_animate_graphs(solution, l, K):
    # Set up the figure with a specified figure size
    fig = plt.figure(figsize=(10, 12))

    # Create a grid layout with 4 rows and 2 columns
    gs = fig.add_gridspec(4, 2)

    # Assign the subplots to the grid
    ax1 = fig.add_subplot(gs[:, 0])  # Inverted pendulum plot spanning all rows
    ax2 = fig.add_subplot(gs[0, 1])  # Theta error plot
    ax3 = fig.add_subplot(gs[1, 1])  # Torque plot
    ax4 = fig.add_subplot(gs[2, 1])  # Angular velocity plot
    ax5 = fig.add_subplot(gs[3, 1])  # Angular acceleration plot

    # Set plot limits and labels
    ax1.set_xlim((-l, l))
    ax1.set_ylim((-l, 1.5 * l))
    ax1.set_aspect('equal')  # Ensure the wheel looks like a circle

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Theta Error (degrees)')

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Torque (Nm)')

    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Angular Velocity (turns/s)')

    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Angular Acceleration (turns/s²)')

    # Initializations for plots
    line, = ax1.plot([], [], 'k-', lw=2)
    wheel_radius = 0.05 * l
    wheel = plt.Circle((0, l), wheel_radius, color='blue', fill=False)
    ax1.add_patch(wheel)

    line_error, = ax2.plot([], [], 'r-', lw=2)
    line_torque, = ax3.plot([], [], 'b-', lw=2)
    line_angular_velocity, = ax4.plot([], [], 'g-', lw=2)
    line_angular_acceleration, = ax5.plot([], [], 'm-', lw=2)

    # Define init function for animation
    def init():
        line.set_data([], [])
        wheel.center = (0, l)
        line_error.set_data([], [])
        line_torque.set_data([], [])
        line_angular_velocity.set_data([], [])
        line_angular_acceleration.set_data([], [])
        return line, wheel, line_error, line_torque, line_angular_velocity, line_angular_acceleration

    # Define animate function
    def animate(i):
        x = [0, l * np.sin(solution.y[0, i])]
        y = [0, -l * np.cos(solution.y[0, i])]
        line.set_data(x, y)
        wheel.center = (x[1], y[1])

        line_error.set_data(solution.t[:i], np.degrees(solution.y[0, :i]))
        line_torque.set_data(solution.t[:i], -K @ solution.y[:, :i])
        line_angular_velocity.set_data(solution.t[:i], solution.y[2, :i] / (2 * np.pi))
        angular_acceleration = np.gradient(solution.y[2, :i], solution.t[:i]) / (2 * np.pi)
        line_angular_acceleration.set_data(solution.t[:i], angular_acceleration)

        return line, wheel, line_error, line_torque, line_angular_velocity, line_angular_acceleration

    # Create the animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(solution.t), interval=20, blit=True)

    return anim, fig


# Example usage
initial_conditions = [np.radians(180), 0, 0]  # Inverted position
Kp, Ki, Kd = 1.0, 0.1, 0.05  # Example PID gains
integral, last_error, dt = 0, 0, 0.1  # Initial PID states

# Wrapper for PID controller
def pid_wrapper(state):
    global integral, last_error
    output, integral, last_error = pid_controller(state, Kp, Ki, Kd, integral, last_error, dt)
    return output

# Choose the controller here
controller = lqr_controller  # or pid_wrapper for PID control

# Run the simulation
solution = simulate_inverted_pendulum(controller, initial_conditions)

# Animate the results
anim, fig = setup_and_animate_graphs(solution, l, place_poles(A, B, np.array([-1, -1.1, -1.2])).gain_matrix)
plt.tight_layout()
plt.show()

