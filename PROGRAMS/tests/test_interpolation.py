import os
import random

import numpy as np

from main_PA2 import compute_C_i_expected, validate_dataset_prefix
from utils.data_processing import dataset_prefixes, parse_calreadings
from utils.interpolation import (
    apply_distortion_correction_bernstein,
    get_distortion_correction_coeffs_bernstein_3d,
)

current_script_path = os.path.abspath(__file__)
CUR_DIR = os.path.dirname(current_script_path)


def test_bernstein_interpolation_1():
    """Tests bernstein interpolation using all available data and
    asserts that the MSE is below a small threshold between the
    data points."""
    for dataset_prefix in dataset_prefixes:
        validate_dataset_prefix(dataset_prefix)  # check if dataset prefix valid
        dataset_folder = os.path.join(CUR_DIR, "..", "..", "DATA")
        calbody_path = os.path.join(dataset_folder, f"{dataset_prefix}calbody.txt")
        calreadings_path = os.path.join(
            dataset_folder, f"{dataset_prefix}calreadings.txt"
        )
        C_i_expected_frames = compute_C_i_expected(calbody_path, calreadings_path)
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
        global_min = np.min(
            np.concatenate([min_expected, min_measured], axis=0), axis=0
        )
        global_max = np.max(
            np.concatenate([max_expected, max_measured], axis=0), axis=0
        )
        # get the coeffs of the bernstein polynomial that approximates the
        # distortion
        degree = 7
        coeffs = get_distortion_correction_coeffs_bernstein_3d(
            gt_pts=C_i_expected,
            measured_pts=C_i_measured,
            degree=degree,
            min=global_min,
            max=global_max,
        )
        # test approximation
        rectified_C_i_measured = apply_distortion_correction_bernstein(
            C_i_measured, coeffs, degree, global_min, global_max
        )
        mse_rectified = np.average((C_i_expected - rectified_C_i_measured) ** 2)
        assert mse_rectified < 0.08
        # print(f"mean abs error rectified - degree({degree}): {mean_abs_error_rectified}.")


def test_bernstein_interpolation_2():
    """Tests bernstein interpolation by breaking up the available data into
    a train/test split. Fitting a Bernstein polynomial to the train data
    and ensuring the error between the rectified measured and gt test values
    is sufficiently small.
    """
    mse_list = list()
    for dataset_prefix in dataset_prefixes:
        validate_dataset_prefix(dataset_prefix)  # check if dataset prefix valid
        dataset_folder = os.path.join(CUR_DIR, "..", "..", "DATA")
        calbody_path = os.path.join(dataset_folder, f"{dataset_prefix}calbody.txt")
        calreadings_path = os.path.join(
            dataset_folder, f"{dataset_prefix}calreadings.txt"
        )
        C_i_expected_frames = compute_C_i_expected(calbody_path, calreadings_path)
        calreadings = parse_calreadings(calreadings_path)
        C_i_measured = np.array([x["C"] for x in calreadings])
        C_i_expected = C_i_expected_frames.reshape(
            C_i_expected_frames.shape[0] * C_i_expected_frames.shape[1],
            C_i_expected_frames.shape[2],
        )
        C_i_measured = C_i_measured.reshape(
            C_i_measured.shape[0] * C_i_measured.shape[1], C_i_measured.shape[2]
        )
        train_size = int(0.8 * len(C_i_expected))
        train_idxs = set(random.sample(range(len(C_i_expected)), train_size))
        test_idxs = [i for i in range(len(C_i_expected)) if i not in train_idxs]
        train_idxs = list(train_idxs)
        C_i_expected_train = C_i_expected[train_idxs]
        C_i_expected_test = C_i_expected[test_idxs]
        C_i_measured_train = C_i_measured[train_idxs]
        C_i_measured_test = C_i_measured[test_idxs]
        min_expected = np.expand_dims(np.min(C_i_expected, axis=0), axis=0)
        max_expected = np.expand_dims(np.max(C_i_expected, axis=0), axis=0)
        min_measured = np.expand_dims(np.min(C_i_measured, axis=0), axis=0)
        max_measured = np.expand_dims(np.max(C_i_measured, axis=0), axis=0)
        global_min = np.min(
            np.concatenate([min_expected, min_measured], axis=0), axis=0
        )
        global_max = np.max(
            np.concatenate([max_expected, max_measured], axis=0), axis=0
        )
        # get the coeffs of the bernstein polynomial that approximates the
        # distortion
        best_degree = None
        min_mse = 10000000
        for degree in range(10):
            coeffs = get_distortion_correction_coeffs_bernstein_3d(
                gt_pts=C_i_expected_train,
                measured_pts=C_i_measured_train,
                degree=degree,
                min=global_min,
                max=global_max,
            )
            # test approximation
            rectified_C_i_measured_test = apply_distortion_correction_bernstein(
                C_i_measured_test, coeffs, degree, global_min, global_max
            )
            mse_rectified = np.average(
                (C_i_expected_test - rectified_C_i_measured_test) ** 2
            )
            if mse_rectified < min_mse:
                best_degree = degree
                min_mse = mse_rectified
            print(mse_rectified)
            # print(f"mean abs error rec
        coeffs = get_distortion_correction_coeffs_bernstein_3d(
            gt_pts=C_i_expected_train,
            measured_pts=C_i_measured_train,
            degree=best_degree,
            min=global_min,
            max=global_max,
        )
        # test approximation
        rectified_C_i_measured_test = apply_distortion_correction_bernstein(
            C_i_measured_test, coeffs, best_degree, global_min, global_max
        )
        mse_rectified = np.average(
            (C_i_expected_test - rectified_C_i_measured_test) ** 2
        )
        mse_list.append(mse_rectified)

    assert np.max(mse_rectified) < 0.5


if __name__ == "__main__":
    test_bernstein_interpolation_1()
