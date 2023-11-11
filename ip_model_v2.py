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

# Modified equations of motion
def equations_of_motion(t, y):
    theta, theta_dot, phi_dot = y  # Assuming torque is not a state variable
    torque = -K_p * theta  # Control law
    d_theta_dot = (m_p * g * l * np.sin(theta) - torque) / (I_p + m_p * l**2)
    d_phi_dot = torque / I_w
    return [theta_dot, d_theta_dot, d_phi_dot]  # Return only state derivatives

# Initial conditions
theta0, theta_dot0, phi_dot0 = np.pi / 6, 0, 0  # Initial conditions for the state variables
initial_conditions = [theta0, theta_dot0, phi_dot0]

# Time span for simulation
t_span, t_eval = [0, 10], np.linspace(0, 10, 300)

# Solve the differential equations
solution = solve_ivp(equations_of_motion, t_span, initial_conditions, t_eval=t_eval)

# Convert angle from radians to degrees for plotting
angle_in_degrees = np.degrees(solution.y[0])

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(solution.t, angle_in_degrees, label='Pendulum Angle (degrees)')
plt.ylabel('Angle (degrees)')
plt.title('Inverted Pendulum with Reaction Wheel')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
torque = -K_p * solution.y[0]  # Torque calculation remains in radians
plt.plot(solution.t, torque, label='Torque (Nm)')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.grid(True)

plt.show()