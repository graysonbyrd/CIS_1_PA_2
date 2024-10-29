import numpy as np
from scipy.special import comb

# Function to compute the Bernstein polynomial basis function
def bernstein_poly(i, n, t):
    return comb(n, i) * (t ** i) * (1 - t) ** (n - i)

# 3D Bernstein polynomial
def bernstein_3d(i, j, k, n, x, y, z):
    return bernstein_poly(i, n, x) * bernstein_poly(j, n, y) * bernstein_poly(k, n, z)

# Sample data for 3D coordinates (observed data points, replace with actual data)
observed_data = np.array([
    [0.1, 0.1, 0.1],
    [0.2, 0.2, 0.15],
    [0.4, 0.3, 0.2],
    # Add more 3D points
])

# Number of control points (degree of Bernstein polynomials)
n = 3

# Generate the grid of control points
control_points = np.random.rand((n+1), (n+1), (n+1), 3)  # Replace with actual control points

# Function to approximate a point using Bernstein polynomials
def approximate_point(x, y, z, control_points, n):
    approx = np.zeros(3)
    for i in range(n+1):
        for j in range(n+1):
            for k in range(n+1):
                bernstein_value = bernstein_3d(i, j, k, n, x, y, z)
                approx += bernstein_value * control_points[i, j, k]
    return approx

# Fit the Bernstein polynomial model to approximate 3D distortion
def fit_bernstein_3d(observed_data, control_points, n):
    # Assuming observed_data is Nx3 where N is number of points and each row is [x, y, z]
    approximations = []
    for point in observed_data:
        x, y, z = point
        approximations.append(approximate_point(x, y, z, control_points, n))
    return np.array(approximations)

# Apply fitting
approximated_points = fit_bernstein_3d(observed_data, control_points, n)

print("Approximated Points:\n", approximated_points)