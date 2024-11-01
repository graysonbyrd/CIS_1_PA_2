from typing import Tuple

import numpy as np

from .transform import FT


def get_pcd_in_local_frame(pcd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Takes a pointcloud in one coordinate frame and transforms that
    pointcloud to a local coordinate frame with origin at the centroid
    of the pointcloud.

    Params:
        pcd (np.ndarray): pointcloud to transform

    Returns:
        np.ndarray: transformed pointcloud in local coordinate frame
        np.ndarray: centroid of pointcloud in original coordinate frame
    """
    if pcd.shape[1] != 3:
        raise ValueError(f"Pcd is size: (n, {pcd.shape[1]}). Must be (n, 3)")
    centroid = np.mean(pcd, axis=0)
    pcd_local = pcd - centroid
    return pcd_local, centroid


def pcd_to_pcd_reg_w_known_correspondence(a: np.ndarray, b: np.ndarray) -> FT:
    """Finds the optimal transform that aligns point cloud a to point cloud b
    using linear least squares via Singular Value Decomposition (SVD).

    Uses SVD shown in:
        https://cdn-uploads.piazza.com/paste/l7e4lajaoxd7hf/c11c58e8338b8842c63d0e51cde943300ad3de876a1c4c05559aa4ed0f668bff/svd1.pdf
        https://www.youtube.com/watch?v=dhzLQfDBx2Q

    Args:
        a (np.array): the target pointcloud
        b (np.array): the source pointcloud

    Returns:
        FT: Frame transform such that the mean squared error between FT(a)
            and b is minimized.
    """
    # align the points in local coordinate frame
    a_local, a_centroid = get_pcd_in_local_frame(a)
    b_local, b_centroid = get_pcd_in_local_frame(b)
    # compute the covariance matrix
    H = np.dot(a_local.T, b_local)
    # Singular Value Decomposition (SVD)
    U, _, Vt = np.linalg.svd(H)
    # compute the optimal rotation matrix
    R = np.dot(Vt.T, U.T)
    # handle reflection case (ensure a proper rotation)
    # explanation shown in part VI of: https://ieeexplore.ieee.org/document/4767965
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    # compute translation vector
    t = np.expand_dims(b_centroid - (R @ a_centroid.T).T, axis=0)
    return FT(R, t)
