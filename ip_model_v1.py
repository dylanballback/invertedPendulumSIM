from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# Physical parameters
m_p = 0.5  # Mass of the pendulum (kg)
l = 0.5    # Length to pendulum's center of mass (m)
I_p = m_p * l**2 / 3  # Moment of inertia of the pendulum (kg.m^2)
m_w = 0.1  # Mass of the reaction wheel (kg)
I_w = 0.005  # Moment of inertia of the reaction wheel (kg.m^2)
g = 9.81  # Gravitational acceleration (m/s^2)

# Control parameters
K_p = 2  # Proportional gain for controller


# Equations of motion
def equations_of_motion(t, y):
    theta, theta_dot, phi_dot = y  # Unpack state variables

    # Control law (simple proportional controller)
    torque = -K_p * theta

    # Equations derived from Lagrange's equation
    d_theta_dot = (m_p * g * l * np.sin(theta) - torque) / (I_p + m_p * l**2)
    d_phi_dot = torque / I_w

    return [theta_dot, d_theta_dot, d_phi_dot]


# Initial conditions
theta0 = np.pi / 6  # Initial angle of the pendulum (30 degrees)
theta_dot0 = 0  # Initial angular velocity of the pendulum
phi_dot0 = 0  # Initial angular velocity of the wheel
initial_conditions = [theta0, theta_dot0, phi_dot0]

# Time span for simulation
t_span = [0, 10]  # Simulate for 10 seconds
t_eval = np.linspace(t_span[0], t_span[1], 300)  # Time points at which to store the solution

# Solve the differential equations
solution = solve_ivp(equations_of_motion, t_span, initial_conditions, t_eval=t_eval)

# Plot results
plt.plot(solution.t, solution.y[0], label='Pendulum Angle (rad)')
plt.plot(solution.t, solution.y[1], label='Pendulum Angular Velocity (rad/s)')
plt.xlabel('Time (s)')
plt.ylabel('State')
plt.title('Inverted Pendulum with Reaction Wheel')
plt.legend()
plt.grid(True)
plt.show()
