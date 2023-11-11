import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import numpy as np
from scipy.signal import place_poles
from scipy.integrate import cumtrapz


# Define the physical parameters (as before)
m_p = 0.5  # Mass of the pendulum (kg)
l = 1.0    # Length to pendulum's center of mass (m)
I_p = m_p * l**2 / 3  # Moment of inertia of the pendulum (kg.m^2)
m_w = 0.1  # Mass of the reaction wheel (kg)
I_w = 0.005  # Moment of inertia of the reaction wheel (kg.m^2)
g = 9.81  # Gravitational acceleration (m/s^2)
K_p = 2  # Proportional gain for controller

# Set the wheel radius for plot
wheel_radius = 0.1 * l  # Set this to an appropriate fraction of the pendulum length


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


# Time span for the simulation
t_span = [0, 10]  # Simulate for 10 seconds

# Solve the differential equations over a time span
initial_conditions = [np.pi / 6, 0, 0]  # Initial conditions

# Solve the differential equations over the defined time span
solution = solve_ivp(equations_of_motion_with_controller, t_span, initial_conditions, t_eval=np.linspace(*t_span, 300))

# Calculate the maximum and minimum theta error and torque for setting plot limits
max_theta_error_degrees = np.max(np.abs(np.degrees(solution.y[0]))) + 10  # Plus 10 degrees buffer
max_torque = np.max(np.abs(-K @ solution.y)) + 2  # Plus 2 Nm buffer

# Calculate the maximum angular velocity for setting plot limits
max_angular_velocity = np.max(np.abs(solution.y[2]))  # Assuming solution.y[2] contains phi_dot

# Set up the figure with a specified figure size
fig = plt.figure(figsize=(10, 12))

# Create a grid layout with 3 rows and 2 columns
# The first column will span all rows for the pendulum animation
# The second column will have three rows for the theta error, torque, and angular velocity plots
gs = fig.add_gridspec(3, 2)

# Assign the subplots to the grid
ax1 = fig.add_subplot(gs[:, 0])  # Inverted pendulum plot spanning all rows
ax2 = fig.add_subplot(gs[0, 1])  # Theta error plot
ax3 = fig.add_subplot(gs[1, 1])  # Torque plot
ax4 = fig.add_subplot(gs[2, 1])  # Angular velocity plot

# Initialize the inverted pendulum plot
ax1.set_xlim((-l, l))
ax1.set_ylim((-l, 1.5 * l))
ax1.set_aspect('equal')  # Ensure the wheel looks like a circle

# Initialize the theta error plot
ax2.set_xlim(t_span)
ax2.set_ylim(-max_theta_error_degrees, max_theta_error_degrees)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Theta Error (degrees)')
line_error, = ax2.plot([], [], 'r-', lw=2)

# Initialize the torque plot
ax3.set_xlim(t_span)
ax3.set_ylim(-max_torque, max_torque)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Torque (Nm)')
line_torque, = ax3.plot([], [], 'b-', lw=2)

# Initialize the angular velocity plot with dynamic limits
ax4.set_xlim(t_span)
ax4.set_ylim(-max_angular_velocity * 1.1, max_angular_velocity * 1.1)  # 10% buffer
ax4.set_ylabel('Angular Velocity (rad/s)')
ax4.set_xlabel('Time (s)')

# The line represents the pendulum, and the circle represents the reaction wheel
line, = ax1.plot([], [], 'k-', lw=2)
wheel = plt.Circle((0, l), wheel_radius, color='blue', fill=False)
ax1.add_patch(wheel)

#Draw a horizontal line at y = 0 for all 3 plots
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1)


# Initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    wheel.center = (0, l)
    line_error.set_data([], [])
    line_torque.set_data([], [])
    return line, wheel, line_error, line_torque

# Initialize lists to store the history of the angle error and torque
theta_error_history = []
torque_history = []

# Assume 'phi_dot' array holds the angular velocities of the reaction wheel
# You would need to calculate this as part of your system state or derive it from the torques
phi_dot = solution.y[2]  # This is an array of angular velocities

# Integrate angular velocity over time to get the angle
phi = cumtrapz(phi_dot, solution.t, initial=0)

# Add a marker on the wheel
wheel_marker, = ax1.plot([], [], 'ro')  # 'ro' creates a red dot


# Initialize a line for the angular velocity plot
line_angular_velocity, = ax4.plot([], [], 'g-', lw=2)  # 'g-' creates a green line

# Animation function which updates figure data. This is called sequentially
def animate(i):
    # Update the pendulum animation
    theta = solution.y[0, i]
    x = l * np.sin(theta)
    y = l * np.cos(theta)
    line.set_data([0, x], [0, y])
    wheel.center = (x, y)
    
    # Update the angle error plot
    theta_error_degrees = np.degrees(solution.y[0, i])  # Current angle error in degrees
    theta_error_history.append(theta_error_degrees)  # Append to the history
    time_points = solution.t[:i+1]  # Time points up to the current frame
    line_error.set_data(time_points, theta_error_history)
    
    # Update the torque plot
    torque = -K @ solution.y[:, i]
    torque_history.append(torque)  # Append to the history
    line_torque.set_data(time_points, torque_history)

    # Update the marker's position based on the rotation angle 'phi'
    marker_angle = phi[i]
    marker_x = wheel.center[0] + wheel_radius * np.cos(marker_angle)
    marker_y = wheel.center[1] + wheel_radius * np.sin(marker_angle)
    wheel_marker.set_data([marker_x], [marker_y])

    # Update the angular velocity plot
    line_angular_velocity.set_data(solution.t[:i+1], solution.y[2, :i+1])
    
    return line, wheel, wheel_marker, line_error, line_torque, line_angular_velocity


# Call the animator. blit=True means only re-draw the parts that have changed.
anim = FuncAnimation(fig, animate, init_func=init, frames=len(solution.t), interval=20, blit=True)

plt.tight_layout()
plt.show()