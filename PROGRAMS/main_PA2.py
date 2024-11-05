import os
from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np

from utils.data_processing import (
    dataset_prefixes,
    parse_calbody,
    parse_calreadings,
    parse_ct_fiducials,
    parse_em_fiducials,
    parse_em_nav,
    parse_empivot,
    parse_output_2,
)
from utils.interpolation import (
    apply_distortion_correction_bernstein,
    get_distortion_correction_coeffs_bernstein_3d,
)
from utils.pcd_2_pcd_reg import (
    get_pcd_in_local_frame,
    pcd_to_pcd_reg_w_known_correspondence,
)
from utils.pivot_cal import pivot_calibration
from utils.transform import FT

current_script_path = os.path.abspath(__file__)
CUR_DIR = os.path.dirname(current_script_path)

# function for question 1
def compute_C_i_expected(calbody_file_path: str, calreadings_file_path: str):
    """Takes in the calibration dataset prefix (e.g. "pa2-debug-c-").
    Loads the relevant data, and follows the procedures outlined
    in Question 4 under Assignment 1 in CIS I PA 1 to compute the C_i_expected
    for each frame in the calibration dataset.

    Params:
        calibration_dataset_prefix (str): the prefix of the calibration
            dataset

    Returns:
        np.ndarray: the computed C_i_expected for each frame in the
            calibration dataset.
    """
    calbody = parse_calbody(calbody_file_path)
    calreadings = parse_calreadings(calreadings_file_path)
    # for each frame, compute C_i_expected
    C_i_expected_frames = list()
    for frame in calreadings:
        d_vals = calbody["d"]
        a_vals = calbody["a"]
        c_vals = calbody["c"]
        D_vals = frame["D"]
        A_vals = frame["A"]
        F_Dd = pcd_to_pcd_reg_w_known_correspondence(d_vals, D_vals)
        F_Aa = pcd_to_pcd_reg_w_known_correspondence(a_vals, A_vals)
        # compute C_i_expected = F_Dd_inv * F_Aa * c_i
        C_i_expected = F_Dd.inverse_transform_pts(F_Aa.transform_pts(c_vals))
        C_i_expected_frames.append(C_i_expected)
    return np.array(C_i_expected_frames)


def get_distortion_calibration_bernstein_polynomial_coeffs(
    calreadings_path: str, C_i_expected_frames: np.ndarray, degree: int
):
    """A helper function that gets the coefficients for a 3D
    Bernstein Polynomial interpolation to calibrate for the
    distortion."""
    calreadings = parse_calreadings(calreadings_path)
    C_i_measured = np.array([x["C"] for x in calreadings])
    C_i_expected = C_i_expected_frames.reshape(
        C_i_expected_frames.shape[0] * C_i_expected_frames.shape[1],
        C_i_expected_frames.shape[2],
    )
    C_i_measured = C_i_measured.reshape(
        C_i_measured.shape[0] * C_i_measured.shape[1], C_i_measured.shape[2]
    )
    min_expected = np.expand_dims(np.min(C_i_expected, axis=0), axis=0)
    max_expected = np.expand_dims(np.max(C_i_expected, axis=0), axis=0)
    min_measured = np.expand_dims(np.min(C_i_measured, axis=0), axis=0)
    max_measured = np.expand_dims(np.max(C_i_measured, axis=0), axis=0)
    global_min = np.min(np.concatenate([min_expected, min_measured], axis=0), axis=0)
    global_max = np.max(np.concatenate([max_expected, max_measured], axis=0), axis=0)
    # apply 5% padding on either side of the global min and global max
    pad_size = (global_max - global_min) * 0.1
    global_min -= pad_size
    global_max += pad_size
    # get the coeffs of the bernstein polynomial that approximates the
    # distortion
    coeffs = get_distortion_correction_coeffs_bernstein_3d(
        gt_pts=C_i_expected,
        measured_pts=C_i_measured,
        degree=degree,
        min=global_min,
        max=global_max,
    )
    return coeffs, global_min, global_max


# function for question 3
def compute_p_tip_in_reference_pointer_frame(
    empivot_file_path: str,
    coeffs: np.ndarray,
    degree: int,
    min: np.ndarray,
    max: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Given an empivot dataset file path, compute the location of p_tip
    with respect to some reference local pointer frame. Return p_tip and
    the reference local pointer frame.

    Params:
        empivot_file_path (str): path to the empivot dataset
        coeffs (np.ndarray): the coefficients for the Bernstein polynomial
        degree (int): the degree of the Bernstein polynomial
        min (np.ndarray): array of 3 values corresponding the x, y, and
            z minimum values for normalization during the distortion
            correction
        max (np.ndarray): array of 3 values corresponding the x, y, and
            z maximum values for normalization during the distortion
            correction

    Returns:
        p_tip (np.ndarray): a numpy array representing the 3d coordinates
            of p_dimple in the em tracker coordinate frame
        reference_ptr_pcd (np.ndarray): reference pointer point cloud
            that p_tip is measured from. the origin of this frame
            is the centroid of the initial point cloud measured in the
            em coordinate frame
    """
    empivot_cal_frames = parse_empivot(empivot_file_path)
    pcd_frames = [x["G"] for x in empivot_cal_frames]
    # for each frame, rectify the pointer point cloud with the distortion
    # calibration
    rectified_pcd_frames = list()
    for pcd in pcd_frames:
        rectified_pcd = apply_distortion_correction_bernstein(
            pcd, coeffs, degree, min, max
        )
        rectified_pcd_frames.append(rectified_pcd)
    p_tip, _, reference_ptr_pcd = pivot_calibration(rectified_pcd_frames)
    return p_tip, reference_ptr_pcd


def compute_p_tip_in_tracker_frame(
    reference_ptr_pcd: np.ndarray,
    pointer_markers_in_tracker_frame: np.ndarray,
    p_tip_in_reference_pointer_frame: np.ndarray,
):
    """Computes the location of p_tip in the tracker frame.

    Args:
        reference_ptr_pcd (np.ndarray): the reference point cloud that
            p_tip is measured from
        pointer_markers_in_tracker_frame (np.ndarray): the locations
            of the pointer markers measured in the tracker frame
        p_tip_in_reference_pointer_frame (np.ndarray): the location
            of p_tip measured from the reference pointer point cloud frame
    """
    pointer_markers_in_local_pointer_frame, _ = get_pcd_in_local_frame(
        pointer_markers_in_tracker_frame
    )
    reference_ptr_pcd_in_local_pointer_frame, _ = get_pcd_in_local_frame(
        reference_ptr_pcd
    )
    # rotate p_tip_local based on the transform between the pointer markers
    # in local pointer frame (taken from the EM frame) and the reference
    # pointer pointcloud
    FT_ptr_orientation = pcd_to_pcd_reg_w_known_correspondence(
        reference_ptr_pcd_in_local_pointer_frame, pointer_markers_in_local_pointer_frame
    )
    p_tip_oriented = FT_ptr_orientation.transform_pts(p_tip_in_reference_pointer_frame)
    FT_pointer_from_tracker = pcd_to_pcd_reg_w_known_correspondence(
        pointer_markers_in_local_pointer_frame,
        pointer_markers_in_tracker_frame,
    )
    p_tip_in_em_frame = FT_pointer_from_tracker.transform_pts(p_tip_oriented)
    return p_tip_in_em_frame


def compute_fiducials_in_em_frame(
    reference_ptr_pcd: np.ndarray,
    em_fiducials_path: str,
    p_tip_in_pointer_frame: np.ndarray,
    coeffs: np.ndarray,
    degree: int,
    min: np.ndarray,
    max: np.ndarray,
) -> np.ndarray:
    """Computes the locations of the fiducials in the EM tracker
    frame.

    Args:
        reference_ptr_pcd (np.ndarray): the reference point cloud that
            p_tip is measured in
        em_fiducials_path (str): path to the em fiducials data file
        p_tip_in_pointer_frame (np.ndarray): location of p_tip in
            the reference pointer frame
        coeffs (np.ndarray): coefficients of the Bernstein polynomial
            used for distortion correction
        degree (int): degree of the Bernstein polynomial used for
            distortion correction
        min (np.ndarray): array of 3 values corresponding the x, y, and
            z minimum values for normalization during the distortion
            correction
        max (np.ndarray): array of 3 values corresponding the x, y, and
            z maximum values for normalization during the distortion
            correction

    Returns:
        em_fiducial_locations (np.ndarray): the location of the
            fiducial points in the EM coordinate frame
    """

    em_fiducial_frames = parse_em_fiducials(em_fiducials_path)
    em_fiducial_locations = list()
    for pointer_marker_pts in em_fiducial_frames:
        # apply distortion correction to the pointer markers
        pointer_marker_pts_rectified = apply_distortion_correction_bernstein(
            pointer_marker_pts, coeffs, degree, min, max
        )
        p_tip_in_em_frame = compute_p_tip_in_tracker_frame(
            reference_ptr_pcd, pointer_marker_pts_rectified, p_tip_in_pointer_frame
        )
        # p_tip is in contact with the fiducial, therefore the
        # location of the fiducial in em coordinate frame is
        # just the location of p_tip in em coordinate frames
        em_fiducial_locations.append(p_tip_in_em_frame)
    return np.concatenate(em_fiducial_locations, axis=0)


def compute_p_tips_in_ct_frame(
    em_nav_frames: List[np.ndarray],
    p_tip_in_reference_pointer_frame: np.ndarray,
    reference_ptr_pcd: np.ndarray,
    FT_reg: FT,
    coeffs: np.ndarray,
    degree: int,
    min: np.ndarray,
    max: np.ndarray,
) -> List[np.ndarray]:
    """Computes the location of p_tip in the CT coordinate frame.

    Args:
        em_nav_frames ( List[np.ndarray] ): the em navigation data frames
        p_tip_in_reference_pointer_frame ( np.ndarray ): the location of
            p_tip with respect to the reference pointer frame that was
            used to compute p_tip during the pivot calibration
        reference_ptr_pcd ( np.ndarray ): the reference point cloud whose
            centroid p_tip is measured from
        FT_reg ( FT ): the frame transform between the EM frame and CT frame
        coeffs ( np.ndarray ): the coefficients for the Bernstein polynomial
            used for distortion correction
        degree (int): degree of the Bernstein polynomial used for
            distortion correction
        min (np.ndarray): array of 3 values corresponding the x, y, and
            z minimum values for normalization during the distortion
            correction
        max (np.ndarray): array of 3 values corresponding the x, y, and
            z maximum values for normalization during the distortion
            correction

    Returns:
        p_tips_in_ct_frame (List[np.ndarra]): the location of p_tip in the
            CT frame for each frame of data
    """
    p_tips_in_ct_frame = list()
    for pointer_marker_pts_em_frame in em_nav_frames:
        pointer_marker_pts_em_frame_rectified = apply_distortion_correction_bernstein(
            pointer_marker_pts_em_frame, coeffs, degree, min, max
        )
        p_tip_em_frame = compute_p_tip_in_tracker_frame(
            reference_ptr_pcd,
            pointer_marker_pts_em_frame_rectified,
            p_tip_in_reference_pointer_frame,
        )
        p_tip_ct_frame = FT_reg.transform_pts(p_tip_em_frame)
        p_tips_in_ct_frame.append(p_tip_ct_frame)
    return p_tips_in_ct_frame


def compute_debug_output_error(
    dataset_prefix: str,
    predicted_output: List[np.ndarray],
    true_debug_output: List[np.ndarray],
):
    """Helper for sanity checking."""
    assert len(predicted_output) == len(true_debug_output)
    predicted_output_np = np.concatenate(predicted_output)
    true_debug_output_np = np.concatenate(true_debug_output)
    mse = np.average((predicted_output_np - true_debug_output_np) ** 2)
    mae = np.average(np.abs(predicted_output_np - true_debug_output_np))
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    # save results file for sanity checking
    save_name = f"{dataset_prefix}result2.txt"
    save_path = os.path.join(CUR_DIR, "..", "OUTPUT", save_name.lower())
    with open(save_path, "w") as file:
        file.write(f"MSE between debug output and predicted: {mse}\n")
        file.write(f"MAE between debug output and predicted: {mae}\n")
    return mae, mse


def validate_dataset_prefix(dataset_prefix: str):
    assert (
        dataset_prefix is not None
    ), f"You must specify a dataset prefix from the following list: {dataset_prefixes}. Alternatively, use argument --full_run to run for each dataset in the DATA folder."
    assert (
        dataset_prefix in dataset_prefixes
    ), f"Invalid dataset prefix: {dataset_prefix}. Valid dataset prefixes are: {dataset_prefixes}."


def write_3d_point_to_file(pt: np.ndarray, file) -> None:
    """Helper file to write a point to a .txt file and return to next line."""
    assert pt.squeeze().shape == (3,)
    pt = pt.squeeze()
    x = round(float(pt[0]), 2)
    y = round(float(pt[1]), 2)
    z = round(float(pt[2]), 2)
    file.write(f"{x},{y},{z}\n")


def get_data_paths(dataset_prefix: str) -> Tuple:
    """Helper function to quickly get all dataset paths."""
    dataset_folder = os.path.join(CUR_DIR, "..", "DATA")
    calbody_path = os.path.join(dataset_folder, f"{dataset_prefix}calbody.txt")
    calreadings_path = os.path.join(dataset_folder, f"{dataset_prefix}calreadings.txt")
    empivot_path = os.path.join(dataset_folder, f"{dataset_prefix}empivot.txt")
    em_fiducials_path = os.path.join(
        dataset_folder, f"{dataset_prefix}em-fiducialss.txt"
    )
    ct_fiducials_path = os.path.join(
        dataset_folder, f"{dataset_prefix}ct-fiducials.txt"
    )
    em_nav_path = os.path.join(dataset_folder, f"{dataset_prefix}EM-nav.txt")
    output2_path = os.path.join(dataset_folder, f"{dataset_prefix}output2.txt")
    return (
        calbody_path,
        calreadings_path,
        empivot_path,
        em_fiducials_path,
        ct_fiducials_path,
        em_nav_path,
        output2_path,
    )


def save_to_output_file(dataset_prefix: str, p_tips_in_ct_frame: List[np.ndarray]):
    """Helper function to quickly save the results to the output file."""
    N_frames = len(p_tips_in_ct_frame)
    save_name = f"{dataset_prefix}OUTPUT2.TXT"
    save_path = os.path.join(CUR_DIR, "..", "OUTPUT", save_name.lower())
    with open(save_path, "w") as file:
        file.write(f"{N_frames},{save_name}\n")
        for p_tip in p_tips_in_ct_frame:
            write_3d_point_to_file(p_tip, file)


def main(dataset_prefix: str):
    """The main script for programming assignment #2. Specify the prefix
    of the data you wish to run for (e.g. pa1-debug-a-)."""
    validate_dataset_prefix(dataset_prefix)  # check if dataset prefix valid
    (
        calbody_path,
        calreadings_path,
        empivot_path,
        em_fiducials_path,
        ct_fiducials_path,
        em_nav_path,
        output2_path,
    ) = get_data_paths(dataset_prefix)
    # high level approach starts here starts here
    C_i_expected_frames = compute_C_i_expected(calbody_path, calreadings_path)
    # get the coeffs for the distortion calibration
    degree = 5
    coeffs, min, max = get_distortion_calibration_bernstein_polynomial_coeffs(
        calreadings_path, C_i_expected_frames, degree
    )
    # compute p_tip in the pointer frame
    p_tip_in_reference_pointer_frame, reference_ptr_pcd = (
        compute_p_tip_in_reference_pointer_frame(empivot_path, coeffs, degree, min, max)
    )
    # compute the location of the fiducials in the em tracker frame
    fiducials_em_frame = compute_fiducials_in_em_frame(
        reference_ptr_pcd,
        em_fiducials_path,
        p_tip_in_reference_pointer_frame,
        coeffs,
        degree,
        min,
        max,
    )
    # compute the location of the fiducials in the CT tracker frame
    fiducials_ct_frame = parse_ct_fiducials(ct_fiducials_path)
    # compute the registration between the EM and CT tracker frame
    FT_reg = pcd_to_pcd_reg_w_known_correspondence(
        fiducials_em_frame, fiducials_ct_frame
    )
    # compute the location of p_tip from the em navigation frames in the CT frame
    em_nav_frames = parse_em_nav(em_nav_path)
    p_tips_in_ct_frame = compute_p_tips_in_ct_frame(
        em_nav_frames,
        p_tip_in_reference_pointer_frame,
        reference_ptr_pcd,
        FT_reg,
        coeffs,
        degree,
        min,
        max,
    )
    save_to_output_file(dataset_prefix, p_tips_in_ct_frame)
    if "debug" in dataset_prefix:
        # print the mse
        output_2_frames = parse_output_2(output2_path)
        mse, mae = compute_debug_output_error(
            dataset_prefix, p_tips_in_ct_frame, output_2_frames
        )
        return mse, mae


def full_run():
    """Runs the main function on every dataset in the DATA folder."""
    for prefix in dataset_prefixes:
        main(prefix)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_prefix", type=str)
    parser.add_argument("--full_run", default=False, action="store_true")
    args = parser.parse_args()
    if args.full_run:
        full_run()
    else:
        main(args.dataset_prefix)
