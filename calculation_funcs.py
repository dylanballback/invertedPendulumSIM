def calculate_moment_of_inertia(mass, diameter, thickness):
    """
    Calculate the moment of inertia for a cylindrical reaction wheel.

    Parameters:
    - mass (kg): Mass of the reaction wheel.
    - diameter (m): Diameter of the reaction wheel.
    - thickness (m): Thickness of the reaction wheel.

    Returns:
    - Moment of inertia of the reaction wheel (kg.m^2).
    """
    
    radius = diameter / 2
    # Moment of inertia for a solid cylinder rotating about its center of mass
    # I = (1/2) * m * r^2 for rotation about its central axis (longitudinal)
    # I = (1/12) * m * (3 * r^2 + h^2) for rotation about its transverse diameter, where h is thickness
    # For a thin disk (thickness << radius), the second term can be neglected.
    # Assuming the reaction wheel is a thin disk, we use the first formula.
    
    moment_of_inertia = 0.5 * mass * radius**2
    return moment_of_inertia

# Example usage:
mass = 0.5  # kg
diameter = 0.1  # m
thickness = 0.01  # m (example value, not used in calculation for a thin disk)

I_w = calculate_moment_of_inertia(mass, diameter, thickness)
print(I_w)