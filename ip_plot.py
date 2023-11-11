import matplotlib.pyplot as plt
import numpy as np

def plot_inverted_pendulum_vertical(theta_degrees):
    """
    Plots a simple 2D representation of an inverted pendulum with a reaction wheel.
    The pendulum is considered 'straight up' at 0 degrees and the reaction wheel torque
    is used to balance it standing up.

    Parameters:
    - theta_degrees: The angle of the pendulum from the upright vertical position in degrees.
                     Should be 0 if the pendulum is perfectly balanced.
    """
    
    # Convert angle to radians for the calculation
    theta = np.radians(theta_degrees)
    
    # Define the pendulum parameters
    length = 1.0  # Length of the pendulum (arbitrary units)
    wheel_radius = 0.1 * length  # Radius of the reaction wheel
    
    # Calculate the position of the pendulum end
    # When theta is 0, pendulum is perfectly vertical
    pendulum_x = length * np.sin(theta)
    pendulum_y = length * np.cos(theta)
    
    # Create the plot
    fig, ax = plt.subplots()
    
    # Plot the pendulum as a line
    ax.plot([0, pendulum_x], [0, pendulum_y], 'k-', lw=2)
    
    # Plot the reaction wheel as a circle
    wheel = plt.Circle((pendulum_x, pendulum_y), wheel_radius, color='blue', fill=False)
    ax.add_patch(wheel)
    
    # Set the aspect of the plot to be equal
    ax.set_aspect('equal')
    
    # Set limits and labels
    ax.set_xlim([-length - wheel_radius -.1, length + wheel_radius +.1])
    ax.set_ylim([-0.1 * length + .1, length + wheel_radius +.1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Add a title
    ax.set_title('Inverted Pendulum with Reaction Wheel')
    
    # Show the grid
    ax.grid(True)
    
    # Display the plot
    plt.show()

# Test the function with a pendulum perfectly balanced (0 degrees from the vertical)
plot_inverted_pendulum_vertical(0)
