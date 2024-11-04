from typing import List, Tuple

import numpy as np
from utils.pcd_2_pcd_reg import (
    get_pcd_in_local_frame,
    pcd_to_pcd_reg_w_known_correspondence,
)

from .transform import FT


def get_A_mat_for_pivot_cal_least_squares(FTs: List[FT]) -> np.ndarray:
    """Given a list of frame transforms, takes N rotation matrices
    of shape (3, 3) and concatenates them in the 0 axis to get a
    matrix of size (N * 3, 3). Next, creates N (-1 * Identity) matrices
    of shape (3, 3) and concatenates them in the 0 axis to get a
    matrix of size (N * 3, 3). Concatenates the complete rotation
    matrix with the complete negative identity matrix to get a matrix of
    shape (3 * N, 2) in the form:

       [[R_1, -I],
        [R_2, -I],
           ...
        [R_N, -I]]

    For performing least squares with pivot calibration.

    Params:
        FTs (List[FT]): list of frame transformations

    Returns:
        np.ndarray: matrix of shape (N * 3, 2) for pivot calibration least
            squares
    """
    R_list = [F.R for F in FTs]
    neg_I_list = [-1 * np.identity(3) for _ in range(len(FTs))]
    R_mat = np.vstack(R_list)
    neg_I_mat = np.vstack(neg_I_list)
    A = np.concatenate([R_mat, neg_I_mat], axis=1)
    return A


def get_b_mat_for_pivot_cal_least_squares(FTs: List[FT]) -> np.ndarray:
    """Given a list of frame transforms, takes N translation
    component of shape (3, 1) and concatenates them to form
    one matrix, b, of shape (N * 3, 1).

    Params:
        FTs (List[FT]): list of frame transformations

    Returns:
        np.ndarray: matrix of shape (N * 3, 1) for pivot calibration least
            squares
    """
    t_list = [-1 * F.t.T for F in FTs]  # make sure you include the negative!!
    b = np.vstack(t_list)
    return b


def run_pivot_least_squares(FTs: List[FT]) -> np.ndarray:
    """Given a list of frame transforms from pointer calibration, solve the
    least squares problem Ax=b for x.

    Params:
        FTs (List[FT]): list of frame transforms from a pivot calibration

    Returns:
        np.ndarray: the solution for x for the least squares problem Ax=b formed
            from the pivot calibration
    """
    A = get_A_mat_for_pivot_cal_least_squares(FTs)
    b = get_b_mat_for_pivot_cal_least_squares(FTs)
    x, _, _, _ = np.linalg.lstsq(A, b)  # compute least squares
    return x


def pivot_calibration(pcd_frames: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs a pivot calibration to determine p_tip in the pointer
    coordinate frame and p_dimple in the tracker coordinate frame.

    Params:
        pcd_frames (List[np.ndarray]): a list of pointer marker pointclouds in
            the tracker coordinate frame

    Returns:
        p_tip (np.ndarray): the 3d location of the pointer tip in the local
            pointer coordinate frame at the orientation of the pcd in the
            first frame of the pcd_frames
        p_dimple (np.ndarray): the 3d location of the pointer tip with
            respect to the tracker coordinate frame. this represents p_dimple,
            since the pointer tip is assumed to be touching p_dimple
        first_frame_pcd (np.ndarray): it is important to return and record this
            information, as you will need to find the transform between this
            orientation and other pointer pcds in order to use the p_tip value
    """
    FTs = list()
    # get pointer markers in starting local coordinate frame
    reference_pcd = pcd_frames[0]
    pcd_local, _ = get_pcd_in_local_frame(reference_pcd)
    for pcd in pcd_frames:
        # compute FT to align pcd in pointer frame with pcd in tracker frame
        FT = pcd_to_pcd_reg_w_known_correspondence(pcd_local, pcd)
        FTs.append(FT)
    x = run_pivot_least_squares(FTs)
    p_tip = x[:3].T  # tip of pointer in local pointer coord frame
    p_dimple = x[3:].T  # tip of pivot dimple in tracker coord frame
    return p_tip, p_dimple, reference_pcd
