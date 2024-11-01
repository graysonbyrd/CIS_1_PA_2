import os

import numpy as np
from main_PA1 import compute_C_i_expected
from scipy.spatial.transform import Rotation as R
from tests.test_utils import TEST_DIR, compute_avg_mse_between_two_pcds
from utils.data_processing import dataset_prefixes, parse_calreadings
from utils.pcd_2_pcd_reg import pcd_to_pcd_reg_w_known_correspondence
from utils.transform import FT


def test_pcd_to_pcd_with_known_correspondence_registration_w_noise():
    """Test the pcd_to_pcd_w_known_correspondence correspondence algo with
    random noise added to pointclouds.

    1. Create a random target pointcloud and random transform.
    2. Compute source pointcloud by transforming the target pointcloud.
    3. Add random noise to the target pointcloud.
    4. Compute the transform via pcd_to_pcd_w_known_correspondence algo.
    5. Check error is below a threshold.
    """
    total_mse = 0  # track mse
    num_test_samples = 50
    for _ in range(num_test_samples):
        # generate a random pointcloud set
        pcd_size = 10
        target = np.random.rand(pcd_size, 3)
        # get a random rotation matrix and translation vector
        R_deg = np.random.randint(0, 360, (1, 3))  # Rot in degrees
        R_random = R.from_euler("xyz", R_deg, degrees=True).as_matrix().squeeze()
        t_random = np.random.randint(0, 100, (1, 3))
        frame_transform = FT(R_random, t_random)
        # create a second pointcloud using the true_R and true_t
        source = frame_transform.transform_pts(target)
        # add random noise to target to simulate real world scenario
        target += np.random.randn(target.shape[0], target.shape[1])
        FT_pred = pcd_to_pcd_reg_w_known_correspondence(target, source)
        pred_source = FT_pred.transform_pts(target)
        total_mse += compute_avg_mse_between_two_pcds(pred_source, source)
    average_mse = total_mse / num_test_samples  # avg mse for any point in the pcd
    assert average_mse < 1


def test_pcd_to_pcd_with_known_correspondence_registration_no_noise():
    """Test the pcd_to_pcd_w_known_correspondence correspondence algo with
    NO random noise added to pointclouds.

    1. Create a random target pointcloud and random transform.
    2. Compute source pointcloud by transforming the target pointcloud.
    3. Compute the transform via pcd_to_pcd_w_known_correspondence algo.
    4. Check error is below a threshold.
    """
    total_mse = 0  # track mse
    num_test_samples = 50
    for _ in range(num_test_samples):
        # generate a random pointcloud set
        pcd_size = 10
        target = np.random.rand(pcd_size, 3)
        # get a random rotation matrix and translation vector
        R_deg = np.random.randint(0, 360, (1, 3))  # Rot in degrees
        R_random = R.from_euler("xyz", R_deg, degrees=True).as_matrix().squeeze()
        t_random = np.random.randint(0, 100, (1, 3))
        frame_transform = FT(R_random, t_random)
        # create a second pointcloud using the true_R and true_t
        source = frame_transform.transform_pts(target)
        FT_pred = pcd_to_pcd_reg_w_known_correspondence(target, source)
        pred_source = FT_pred.transform_pts(target)
        total_mse += compute_avg_mse_between_two_pcds(pred_source, source)
    average_mse = total_mse / num_test_samples  # avg mse for any point in the pcd
    assert np.isclose(0, average_mse)


def test_compute_C_i_expected():
    """Takes in the calibration dataset prefix (e.g. "pa1-debug-c-").
    Loads the relevant data, and follows the procedures outlined
    in Question 4 under Assignment 1 in CIS I PA 1.

    Params:
        calibration_dataset_prefix (str): the prefix of the calibration
            dataset

    Returns:
        np.ndarray: the computed C_i_expected for each frame in the
            calibration dataset.
    """
    for idx, prefix in enumerate(dataset_prefixes):
        calbody_path = os.path.join(
            TEST_DIR, "..", "..", "DATA", f"{prefix}calbody.txt"
        )
        calreadings_path = os.path.join(
            TEST_DIR, "..", "..", "DATA", f"{prefix}calreadings.txt"
        )
        C_i_expected_frames = compute_C_i_expected(calbody_path, calreadings_path)
        calreadings = parse_calreadings(calreadings_path)
        for idx, C_i_expected in enumerate(C_i_expected_frames):
            C_i_true = calreadings[idx]["C"]
            avg_mse = compute_avg_mse_between_two_pcds(C_i_expected, C_i_true)
            # this number is somewhat arbitrary, but provides a sanity check
            assert avg_mse < 5
