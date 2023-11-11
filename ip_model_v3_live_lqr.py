import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import numpy as np
from scipy.signal import place_poles

# Define the physical parameters (as before)
m_p = 0.5  # Mass of the pendulum (kg)
l = 1.0    # Length to pendulum's center of mass (m)
I_p = m_p * l**2 / 3  # Moment of inertia of the pendulum (kg.m^2)
m_w = 0.1  # Mass of the reaction wheel (kg)
I_w = 0.005  # Moment of inertia of the reaction wheel (kg.m^2)
g = 9.81  # Gravitational acceleration (m/s^2)
K_p = 2  # Proportional gain for controller

# Define the linearized system matrices around the upright position
A = np.array([[0, 1, 0],
              [m_p*g*l/(I_p + m_p*l**2), 0, 0],
              [0, 0, 0]])
B = np.array([[0],
              [-1/(I_p + m_p*l**2)],
              [1/I_w]])
C = np.identity(3)
D = np.array([[0], [0], [0]])

# Define desired pole locations for the controller
# These values would typically be chosen based on the desired response characteristics
poles = np.array([-1, -1.1, -1.2])

# Compute the state feedback matrix
K = place_poles(A, B, poles).gain_matrix

# Modify the equations of motion to use the state feedback controller
def equations_of_motion_with_controller(t, y):
    theta, theta_dot, phi_dot = y
    # State vector
    x = np.array([[theta], [theta_dot], [phi_dot]])
    # Control law: u = -Kx
    torque = -K.dot(x)[0, 0]
    d_theta_dot = (m_p * g * l * np.sin(theta) - torque) / (I_p + m_p * l**2)
    d_phi_dot = torque / I_w
    return [theta_dot, d_theta_dot, d_phi_dot]

# Solve the differential equations over a time span
t_span = [0, 10]  # Time span for simulation
t_eval = np.linspace(*t_span, 300)  # Time points where the solution is computed
initial_conditions = [np.pi / 6, 0, 0]  # Initial conditions
solution = solve_ivp(equations_of_motion_with_controller, t_span, initial_conditions, t_eval=t_eval)

# Set up the figure, the axis, and the plot element we want to animate
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim((-l, l))
ax.set_ylim((-l, 1.5 * l))

# The line represents the pendulum, and the circle represents the reaction wheel
line, = ax.plot([], [], 'k-', lw=2)
wheel_radius = 0.1 * l
wheel = plt.Circle((0, l), wheel_radius, color='blue', fill=False)
ax.add_patch(wheel)

# Initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    wheel.center = (0, l)
    return line, wheel

# Animation function which updates figure data. This is called sequentially
def animate(i):
    theta = solution.y[0, i]
    x = l * np.sin(theta)
    y = l * np.cos(theta)
    line.set_data([0, x], [0, y])
    wheel.center = (x, y)
    return line, wheel

# Call the animator. blit=True means only re-draw the parts that have changed.
anim = FuncAnimation(fig, animate, init_func=init, frames=len(solution.t), interval=20, blit=True)

plt.show()
