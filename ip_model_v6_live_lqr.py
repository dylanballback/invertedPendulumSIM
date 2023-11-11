import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import numpy as np
from scipy.signal import place_poles
from scipy.integrate import cumtrapz
import time

def simulate_inverted_pendulum(initial_theta_degrees, simulation_duration=10):
    """
    Simulate an inverted pendulum with a reaction wheel and animate the results.
    
    Parameters:
    - initial_theta_degrees: The initial angle of the pendulum from the vertical in degrees.
    - simulation_duration: Duration of the simulation in seconds.
    """
    # Convert the initial angle to radians
    initial_theta = np.radians(initial_theta_degrees)

    # Define the physical parameters (as before)
    m_p = 0.5  # Mass of the pendulum (kg)
    l = 1.0    # Length to pendulum's center of mass (m)
    I_p = m_p * l**2 / 3  # Moment of inertia of the pendulum (kg.m^2)
    I_w = 0.005  # Moment of inertia of the reaction wheel (kg.m^2)
    g = 9.81  # Gravitational acceleration (m/s^2)



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

    # Initial conditions for the simulation
    initial_conditions = [initial_theta, 0, 0]  # [theta, theta_dot, phi_dot]

    # Time span for the simulation
    t_span = [0, simulation_duration]
    t_eval = np.linspace(*t_span, 300)

    # Solve the differential equations over the defined time span
    solution = solve_ivp(equations_of_motion_with_controller, t_span, initial_conditions, t_eval=t_eval)


    # Calculate the maximum and minimum theta error and torque for setting plot limits
    max_theta_error_degrees = np.max(np.abs(np.degrees(solution.y[0]))) + 10  # Plus 10 degrees buffer
    max_torque = np.max(np.abs(-K @ solution.y)) + 2  # Plus 2 Nm buffer

    # Calculate the maximum angular velocity in turns per second for setting plot limits
    max_angular_velocity_turns = np.max(np.abs(solution.y[2])) / (2 * np.pi)  # Convert rad/s to turns/s

    # Calculate angular acceleration using numpy.gradient or numpy.diff
    angular_velocity = solution.y[2]  # This is your angular velocity data
    time_points = solution.t  # This is your time data

    # Calculate angular acceleration using numpy.gradient to compute the derivative of angular velocity
    angular_acceleration_rads = np.gradient(angular_velocity, time_points)

    # Convert angular acceleration to turns/s²
    angular_acceleration_turns = angular_acceleration_rads / (2 * np.pi)

    # Using numpy.gradient to compute the derivative of angular velocity
    angular_acceleration = np.gradient(angular_velocity, time_points)

    # Calculate the maximum and minimum angular acceleration values for setting plot limits
    max_angular_acceleration = np.max(angular_acceleration_turns)
    min_angular_acceleration = np.min(angular_acceleration_turns)

    # Define the margin you want around the max and min values for the y-axis
    margin = 20  # This is your desired margin

    # Set the wheel radius for plot
    wheel_radius = 0.1 * l  # Set this to an appropriate fraction of the pendulum length

    # Set up the figure with a specified figure size
    fig = plt.figure(figsize=(10, 12))

    # Create a grid layout with 3 rows and 2 columns
    # The first column will span all rows for the pendulum animation
    # The second column will have three rows for the theta error, torque, and angular velocity plots
    gs = fig.add_gridspec(4, 2)

    # Assign the subplots to the grid
    ax1 = fig.add_subplot(gs[:, 0])  # Inverted pendulum plot spanning all rows
    ax2 = fig.add_subplot(gs[0, 1])  # Theta error plot
    ax3 = fig.add_subplot(gs[1, 1])  # Torque plot
    ax4 = fig.add_subplot(gs[3, 1])  # Angular velocity plot
    ax5 = fig.add_subplot(gs[2, 1])  # Add a new subplot for angular acceleration

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
    ax4.set_ylim(-max_angular_velocity_turns * 1.1, max_angular_velocity_turns * 1.1)  # 10% buffer
    ax4.set_ylabel('Angular Velocity (turns/s)')
    ax4.set_xlabel('Time (s)')

    #Initalize the angular acceleration plot 
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Angular Acceleration (turns/s²)')
    ax5.set_xlim(t_span)
    ax5.set_ylim(min_angular_acceleration - margin, max_angular_acceleration + margin)
    line_angular_acceleration, = ax5.plot([], [], 'm-', lw=2)  # 'm-' creates a magenta line

    


    # The line represents the pendulum, and the circle represents the reaction wheel
    line, = ax1.plot([], [], 'k-', lw=2)
    wheel = plt.Circle((0, l), wheel_radius, color='blue', fill=False)
    ax1.add_patch(wheel)

    #Draw a horizontal line at y = 0 for all 3 plots
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax5.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    # Initialize lists to store the history of the angle error, torque, and angular velocity
    theta_error_history = []
    torque_history = []
    angular_velocity_history = []
    angular_acceleration_history = []

    # Initialize a line for the angular velocity plot
    line_angular_velocity, = ax4.plot([], [], 'g-', lw=2)  # 'g-' creates a green line

    # Assume 'phi_dot' array holds the angular velocities of the reaction wheel
    # You would need to calculate this as part of your system state or derive it from the torques
    phi_dot = solution.y[2]  # This is an array of angular velocities

    # Integrate angular velocity over time to get the angle
    phi = cumtrapz(phi_dot, solution.t, initial=0)

    # Add a marker on the wheel
    wheel_marker, = ax1.plot([], [], 'ro')  # 'ro' creates a red dot

    # Before defining init and animate functions, declare the text objects globally
    text_theta_error = None
    text_torque = None
    text_angular_velocity = None
    text_angular_acceleration = None
    
    # Initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        wheel.center = (0, l)
        wheel_marker.set_data([], [])
        line_error.set_data([], [])
        line_torque.set_data([], [])
        line_angular_velocity.set_data([], [])
        line_angular_acceleration.set_data([], [])
        #Calculate max values and convert angular velocity to turns per second
        max_theta_error = np.max(np.abs(np.degrees(solution.y[0])))
        max_torque_val = np.max(np.abs(-K @ solution.y))
        max_angular_velocity_val = np.max(np.abs(solution.y[2])) / (2 * np.pi)  # Convert rad/s to turns/s

        # Display the max values on the plots
        ax2.text(0.95, 0.95, f'Max Theta Error: {max_theta_error:.2f}°',
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax2.transAxes)

        ax3.text(0.95, 0.95, f'Max Torque: {max_torque_val:.2f} Nm',
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax3.transAxes)

        ax4.text(0.95, 0.95, f'Max Angular Velocity: {max_angular_velocity_val:.2f} turns/s',
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax4.transAxes)
        
        ax5.text(0.95, 0.95, f'Max Angular Acceleration: {max_angular_acceleration:.2f} turns/s^2',
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax5.transAxes)
        
        # Initialize text objects for current values
        global text_theta_error, text_torque, text_angular_velocity, text_angular_acceleration
        # Create text objects for current values, initially empty
        text_theta_error = ax2.text(0.05, 0.95, '', horizontalalignment='right', verticalalignment='top', transform=ax2.transAxes, va='top')
        text_torque = ax3.text(0.05, 0.95, '', horizontalalignment='right', verticalalignment='top', transform=ax3.transAxes, va='top')
        text_angular_velocity = ax4.text(0.05, 0.95, '', horizontalalignment='right', verticalalignment='top', transform=ax4.transAxes, va='top')
        
        # Create text objects for current angular acceleration, initially empty
        text_angular_acceleration = ax5.text(0.95, 0.05, '', horizontalalignment='right', verticalalignment='bottom', transform=ax5.transAxes, va='top')

        
        # Return all the plot elements that need to be initialize
        return (line, wheel, wheel_marker, line_error, line_torque, 
            line_angular_velocity, line_angular_acceleration, text_theta_error, 
            text_torque, text_angular_velocity, text_angular_acceleration)




    

    # Define the delay in terms of the number of frames
    delay_seconds = 3
    frame_rate = 40  # This is the interval in milliseconds (ms)
    delay_frames = int((delay_seconds * 1000) / frame_rate)

    # Animation function which updates figure data. This is called sequentially
    def animate(i):
        global text_theta_error, text_torque, text_angular_velocity, text_angular_acceleration

        # Compute the initial position of the pendulum and the wheel marker
        initial_theta = initial_conditions[0]
        initial_x = l * np.sin(initial_theta)
        initial_y = l * np.cos(initial_theta)
        initial_marker_angle = 0  # Assuming the wheel marker starts at angle 0

        # During the delay, initialize the plots
        if i < delay_frames:
            # Set the pendulum line to the initial position
            line.set_data([0, initial_x], [0, initial_y])
            # Set the wheel to the initial position
            wheel.center = (initial_x, initial_y)
            # Set the wheel marker to the initial position
            initial_marker_x = initial_x + wheel_radius * np.cos(initial_marker_angle)
            initial_marker_y = initial_y + wheel_radius * np.sin(initial_marker_angle)
            wheel_marker.set_data([initial_marker_x], [initial_marker_y])
            # Set initial data for all other plots
            line_error.set_data([], [])
            line_torque.set_data([], [])
            line_angular_velocity.set_data([], [])
            line_angular_acceleration.set_data([], [])
            # Text objects are initialized with empty strings
            text_theta_error.set_text('')
            text_torque.set_text('')
            text_angular_velocity.set_text('')
            text_angular_acceleration.set_text('')
            
    
        else:
            # Subtract the delay to get the correct index in the simulation data
            adjusted_index = i - delay_frames
            
            # Update pendulum and wheel positions
            theta = solution.y[0, adjusted_index]
            x = l * np.sin(theta)
            y = l * np.cos(theta)
            line.set_data([0, x], [0, y])
            wheel.center = (x, y)
            
            # Update the reaction wheel marker to simulate spinning
            marker_angle = phi[adjusted_index]
            marker_x = wheel.center[0] + wheel_radius * np.cos(marker_angle)
            marker_y = wheel.center[1] + wheel_radius * np.sin(marker_angle)
            wheel_marker.set_data([marker_x], [marker_y])
            
            # Append new data to the histories and update the plots
            current_theta_error = np.degrees(solution.y[0, adjusted_index])
            current_torque = -K @ solution.y[:, adjusted_index]
            current_angular_velocity = solution.y[2, adjusted_index] / (2 * np.pi)  # Convert rad/s to turns/s
            current_angular_acceleration = angular_acceleration_turns[adjusted_index]

            # Update the text objects with the current values at the bottom right
            text_theta_error.set_text(f'Theta Error: {current_theta_error:.2f}°')
            text_torque.set_text(f'Torque: {current_torque[0]:.2f} Nm')
            text_angular_velocity.set_text(f'Angular Velocity: {current_angular_velocity:.2f} turns/s')
            text_angular_acceleration.set_text(f'Angular Acceleration: {current_angular_acceleration:.2f} turns/s²')
            text_theta_error.set_position((0.95, 0.05))  # Move text to bottom right
            text_torque.set_position((0.95, 0.05))      # Move text to bottom right
            text_angular_velocity.set_position((0.95, 0.05))  # Move text to bottom right
            text_angular_acceleration.set_position((0.95, 0.05))  # Move text to bottom right

            # Now update the lines with history for error, torque, and angular velocity
            theta_error_history.append(current_theta_error)
            torque_history.append(current_torque)
            angular_velocity_history.append(current_angular_velocity)
            angular_acceleration_history.append(current_angular_acceleration)

            line_error.set_data(solution.t[:adjusted_index+1], theta_error_history)
            line_torque.set_data(solution.t[:adjusted_index+1], torque_history)
            line_angular_velocity.set_data(solution.t[:adjusted_index+1], angular_velocity_history)
            line_angular_acceleration.set_data(time_points[:adjusted_index+1], angular_acceleration_history)
    
        # Return the updated line objects and text objects
        return (line, wheel, wheel_marker, line_error, line_torque, 
            line_angular_velocity, line_angular_acceleration, text_theta_error, 
            text_torque, text_angular_velocity, text_angular_acceleration)

    # Create the FuncAnimation object
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(solution.t) + delay_frames, interval=20, blit=True)
    
    # Save the animation if needed
    #anim.save('inverted_pendulum.mp4')


    return anim, fig


# Example usage: simulate with an initial angle of 30 degrees from the vertical
# Example usage:
anim, fig = simulate_inverted_pendulum(180)

# To prevent the animation from being garbage-collected, anim must be returned from the function
# and stored in a variable that persists (like done above with anim, fig = simulate_inverted_pendulum(30))

plt.tight_layout()
plt.show()