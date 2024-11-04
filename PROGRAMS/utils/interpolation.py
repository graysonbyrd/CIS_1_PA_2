import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lstsq
from scipy.special import comb


def normalize_pcd(pcd: np.ndarray, min: np.ndarray, max: np.ndarray) -> np.ndarray:
    """Normalize a pointcloud between the provided minimum and maximum.

    Args:
        pcd (np.ndarray): the point cloud to normalize
        min (np.ndarray): the minimum value for the normalization in the
            x, y, and z axis
        max (np.ndarray): the maximum value for the normalization in the
            x, y, and z axis

    Returns:
        normalized_pcd (np.ndarray): the normalized pointcloud
    """
    num_axes = pcd.shape[1]
    normalized_pcd = np.zeros_like(pcd)
    for i in range(num_axes):
        normalized_pcd[:, i] = (pcd[:, i] - min[i]) / (max[i] - min[i])
    return normalized_pcd


def denormalize_pcd(pcd: np.ndarray, min: np.ndarray, max: np.ndarray) -> np.ndarray:
    """Denormalize a pointcloud between the provided minimum and maximum.

    Args:
        pcd (np.ndarray): the point cloud to denormalize
        min (np.ndarray): the minimum value for the denormalization in the
            x, y, and z axis
        max (np.ndarray): the maximum value for the denormalization in the
            x, y, and z axis

    Returns:
        denormalized_pcd (np.ndarray): the denormalized pointcloud
    """
    num_axes = pcd.shape[1]
    denormalized_pcd = np.zeros_like(pcd)
    for i in range(num_axes):
        denormalized_pcd[:, i] = pcd[:, i] * (max[i] - min[i]) + min[i]
    return denormalized_pcd


def bernstein_poly(n, i, x):
    """Compute Bernstein polynomial value B_i^n at x."""
    return comb(n, i) * (x**i) * ((1 - x) ** (n - i))


def construct_approximation_matrix(pcd_norm: np.ndarray, degree: int) -> np.ndarray:
    """Computes the approximation matrix, A, for the Bernstein
    Polynomial Interpolation Least Squares Calculation.

    Args:
        pcd_norm (np.ndarray): Normalized distorted point cloud
        degree (int): degree of the Bernstein Polynomial

    Returns:
        A (np.ndarray): the approximation matrix, A, for the Bernstein
        Polynomial Interpolation Least Squares calculation
    """
    # ensure the point cloud is already normalized
    assert np.min(pcd_norm) >= 0 and np.min(pcd_norm) < 1
    assert np.max(pcd_norm) <= 1 and np.max(pcd_norm) > 0
    num_samples = pcd_norm.shape[0]
    x_vals = pcd_norm[:, 0]
    y_vals = pcd_norm[:, 1]
    z_vals = pcd_norm[:, 2]
    # F = B_{n,i}(u_x)*B_{n,i}(u_y)*B_{n,i}(u_z)
    # A should be of size (num_samples, (degree+1)**3)
    A = np.zeros((num_samples, (degree + 1) ** 3))
    count = 0
    for i in range(degree + 1):
        for j in range(degree + 1):
            for k in range(degree + 1):
                A[:, count] = (
                    bernstein_poly(degree, i, x_vals)
                    * bernstein_poly(degree, j, y_vals)
                    * bernstein_poly(degree, k, z_vals)
                )
                count += 1
    return A


def get_distortion_correction_coeffs_bernstein_3d(
    gt_pts: np.ndarray,
    measured_pts: np.ndarray,
    degree: int,
    min: np.ndarray,
    max: np.ndarray,
) -> np.ndarray:
    """Computes the coefficients for the Bernstein Polynomial
    that interpolates to calibrate for the distortion in measured
    points.

    Args:
        gt_pts (np.ndarray): ground truth point cloud points without
            distortion
        measured_pts (np.ndarray): measured point cloud points with
            distortion
        degree (int): the degree of the Bernstein polynomial
        min (np.ndarray): the minimum value for the denormalization in the
            x, y, and z axis
        max (np.ndarray): the maximum value for the denormalization in the
            x, y, and z axis

    coeffs (np.ndarray): the coefficients of the estimated Bernstein
        polynomial that can be used to correct for the distortion in the
        measured point cloud frame
    """
    # normalize the ground truth points
    gt_pts_norm = normalize_pcd(gt_pts, min, max)
    measured_pts_norm = normalize_pcd(measured_pts, min, max)
    A = construct_approximation_matrix(measured_pts_norm, degree)
    coeffs, _, _, _ = np.linalg.lstsq(A, gt_pts_norm)
    return coeffs


def apply_distortion_correction_bernstein(
    measured_points: np.ndarray,
    coeffs: np.ndarray,
    degree: int,
    min: np.ndarray,
    max: np.ndarray,
) -> np.ndarray:
    """Takes measured points and parameters of a Bernstein polynomial
    and rectified those measured points by removing distortion.

    Args:
        measured_points (np.ndarray): measured, distorted point cloud
        coeffs (np.ndarray): the coefficients of the Bernstein
            polynomial for correcting the distortion
        degree (int): the degree of the Bernstein polynomial
        min (np.ndarray): the minimum value for the denormalization in the
            x, y, and z axis
        max (np.ndarray): the maximum value for the denormalization in the
            x, y, and z axis

    Returns:
        rectified_points (np.ndarray): the measured points corrected for
            the distortion in the measured points tracker
    """
    assert measured_points.shape[1] == 3
    assert (degree + 1) ** 3 == coeffs.shape[0]
    measured_points_norm = normalize_pcd(measured_points, min, max)
    A = construct_approximation_matrix(measured_points_norm, degree)
    rectified_points_norm = A @ coeffs
    rectified_points = denormalize_pcd(rectified_points_norm, min, max)
    return rectified_points


def visual_sanity_check_3d_bernstein_approximation():
    degree = 3
    # Generate synthetic ground truth 3D points (e.g., a spiral in 3D)
    t = np.linspace(0, 4 * np.pi, 100)
    x = np.sin(t)
    y = np.cos(t)
    z = t / (4 * np.pi)
    gt_points = np.vstack((x, y, z)).T  # Shape (100, 3)
    min_val = np.array([-1, -1, -1])
    max_val = np.array([1, 1, 1])

    # Normalize the generated points
    gt_points_norm = normalize_pcd(gt_points, min_val, max_val)

    # Get 3D Bernstein polynomial coefficients for the approximation
    coeffs = get_distortion_correction_coeffs_bernstein_3d(
        gt_points, degree, min_val, max_val
    )

    # Prepare for the polynomial approximation using the computed coefficients
    num_samples = gt_points.shape[0]
    x_vals = gt_points_norm[:, 0]
    y_vals = gt_points_norm[:, 1]
    z_vals = gt_points_norm[:, 2]
    A = np.zeros((num_samples, (degree + 1) ** 3))

    # Construct the approximation matrix A as in the main function
    count = 0
    for i in range(degree + 1):
        for j in range(degree + 1):
            for k in range(degree + 1):
                A[:, count] = (
                    bernstein_poly(degree, i, x_vals)
                    * bernstein_poly(degree, j, y_vals)
                    * bernstein_poly(degree, k, z_vals)
                )
                count += 1

    # Compute the approximated points
    approx_points_norm = A @ coeffs
    approx_points = denormalize_pcd(approx_points_norm, min_val, max_val)

    # Plot the original and approximated points for visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot original points in blue
    ax.scatter(
        gt_points[:, 0],
        gt_points[:, 1],
        gt_points[:, 2],
        c="blue",
        label="Original Points",
    )

    # Plot approximated points in red
    ax.scatter(
        approx_points[:, 0],
        approx_points[:, 1],
        approx_points[:, 2],
        c="red",
        marker="^",
        label="Approximated Points",
    )

    ax.set_title(f"3D Bernstein Polynomial Approximation (Degree: {degree})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    visual_sanity_check_3d_bernstein_approximation()

