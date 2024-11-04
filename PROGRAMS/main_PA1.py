import os
from argparse import ArgumentParser
from typing import Tuple

import numpy as np

from utils.data_processing import (
    dataset_prefixes,
    parse_calbody,
    parse_calreadings,
    parse_empivot,
    parse_optpivot,
)
from utils.pcd_2_pcd_reg import pcd_to_pcd_reg_w_known_correspondence
from utils.pivot_cal import pivot_calibration

current_script_path = os.path.abspath(__file__)
CUR_DIR = os.path.dirname(current_script_path)


def get_N_c_and_N_frames_from_calreadings(
    calreadings_file_path: str,
) -> Tuple[int, int]:
    """Helper function to retrieve the N_C and N_frames from a calreadings
    dataset file."""
    frames = parse_calreadings(calreadings_file_path)
    N_frames = len(frames)
    N_C = len(frames[0]["C"])
    return N_C, N_frames


# function for question 4
def compute_C_i_expected(calbody_file_path: str, calreadings_file_path: str):
    """Takes in the calibration dataset prefix (e.g. "pa1-debug-c-").
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


# function for question 5
def compute_p_dimple_in_em_coord_frame(empivot_file_path: str) -> np.ndarray:
    """Given an empivot dataset file path, compute the location of p_dimple
    in the em tracker coord frame.

    Params:
        empivot_file_path (str): path to the empivot dataset

    Returns:
        np.ndarray: a numpy array representing the 3d coordinates of p_dimple
            in the em tracker coordinate frame
    """
    empivot_cal_frames = parse_empivot(empivot_file_path)
    pcd_frames = [x["G"] for x in empivot_cal_frames]
    p_tip, p_dimple, _ = pivot_calibration(pcd_frames)
    return p_dimple


# function for question 6
def compute_p_dimple_in_opt_coord_frame(
    calbody_file_path: str, optpivot_file_path: str
) -> np.ndarray:
    """Given an optpivot dataset file path, compute the location of p_dimple
    in the opt tracker coord frame.

    Params:
        optpivot_file_path (str): path to the optpivot dataset

    Returns:
        np.ndarray: a numpy array representing the 3d corrdinates of p_dimple
            in the em tracker coordinate frame
    """
    calbody = parse_calbody(calbody_file_path)
    optpivot_cal_frames = parse_optpivot(optpivot_file_path)
    # get the H coordinates in the em coordinate frame
    d_vals = calbody["d"]
    H_in_em_pcd_frames = list()
    for frame in optpivot_cal_frames:
        D_vals = frame["D"]
        H_vals = frame["H"]
        F_dD = pcd_to_pcd_reg_w_known_correspondence(D_vals, d_vals)
        H_in_em_pcd_frames.append(F_dD.transform_pts(H_vals))
    p_tip, p_dimple, _ = pivot_calibration(H_in_em_pcd_frames)
    return p_dimple


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
    file.write(f"{x},\t{y},\t{z}\n")


def main(dataset_prefix: str):
    """The main script for programming assignment # 1. Specify the prefix
    of the data you wish to run for (e.g. pa1-debug-a-)."""
    validate_dataset_prefix(dataset_prefix)  # check if dataset prefix valid
    dataset_folder = os.path.join(CUR_DIR, "..", "DATA")
    output_folder = os.path.join(CUR_DIR, "..", "OUTPUT")
    calbody_path = os.path.join(dataset_folder, f"{dataset_prefix}calbody.txt")
    calreadings_path = os.path.join(dataset_folder, f"{dataset_prefix}calreadings.txt")
    empivot_path = os.path.join(dataset_folder, f"{dataset_prefix}empivot.txt")
    optpivot_path = os.path.join(dataset_folder, f"{dataset_prefix}optpivot.txt")
    C_i_expected_frames = compute_C_i_expected(calbody_path, calreadings_path)
    p_dimple_em = compute_p_dimple_in_em_coord_frame(empivot_path)
    p_dimple_opt = compute_p_dimple_in_opt_coord_frame(calbody_path, optpivot_path)
    output_file = os.path.join(output_folder, f"{dataset_prefix}output1.txt")
    N_C, N_frames = get_N_c_and_N_frames_from_calreadings(calreadings_path)
    with open(output_file, "w") as file:
        file.write(f"{N_C}, {N_frames}, {dataset_prefix}output1.txt\n")
        write_3d_point_to_file(p_dimple_em, file)
        write_3d_point_to_file(p_dimple_opt, file)
        for frame in C_i_expected_frames:
            for C_i_expected in frame:
                write_3d_point_to_file(C_i_expected, file)


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
