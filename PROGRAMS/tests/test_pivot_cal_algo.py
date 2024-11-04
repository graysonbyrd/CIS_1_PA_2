import os

import numpy as np
from scipy.spatial.transform import Rotation as R
from tests.test_utils import TEST_DIR, compute_avg_mse_between_two_pcds
from utils.data_processing import dataset_prefixes, parse_empivot
from utils.pcd_2_pcd_reg import (
    get_pcd_in_local_frame,
    pcd_to_pcd_reg_w_known_correspondence,
)
from utils.pivot_cal import pivot_calibration, run_pivot_least_squares
from utils.transform import FT


def test_pivot_cal_w_custom_data():
    """Test the pivot calibration with custom data. Test method is as
    follows:

    1. Define test values for p_tip and p_dimple
    2. For k = 1 to N
        Get random R_k
        Set p_k to p_dimple - R_k dot p_tip
        Set F_k = [R_k, p_k]

    Attempt to recover p_tip and p_dimple using the pivot cal method.
    """
    p_tip = np.array([[-12.321, -42.124, 13.523]])  # test value for p_tip
    p_dimple = np.array([[142.342, 542.432, 1444.479]])  # test value fof p_dimple
    num_data_samples = 10
    FTs = list()
    for _ in range(num_data_samples):
        R_deg = np.random.randint(0, 360, (1, 3))  # Rot in degrees
        R_random = R.from_euler("xyz", R_deg, degrees=True).as_matrix().squeeze()
        t = p_dimple - (R_random @ p_tip.T).T
        F = FT(R_random, t)
        FTs.append(F)
    # get predicted values for p_tip and p_dimple
    x = run_pivot_least_squares(FTs)
    p_tip_pred = x[:3].T
    p_dimple_pred = x[3:].T
    # assert the mse between true and predicted values are small
    mse_p_tip = compute_avg_mse_between_two_pcds(p_tip_pred, p_tip)
    mse_p_dimple = compute_avg_mse_between_two_pcds(p_dimple_pred, p_dimple)
    # assert np.isclose(0, mse_p_tip)
    assert np.isclose(0, mse_p_dimple)


def test_pivot_cal_w_pa_1_data():
    """Sanity check test on the actual data.

    Go through each of the datasets and perform a pivot calibration to
    find p_tip and p_dimple values. For each F_k, ensure there is
    minimal error between F_k(p_tip_pred) and p_dimple_pred.
    """
    total_mse = 0
    avg_mse = 0
    for _, prefix in enumerate(dataset_prefixes):
        data_path = os.path.join(TEST_DIR, "..", "..", "DATA", f"{prefix}empivot.txt")
        cal_frames = parse_empivot(data_path)
        pcd_frames = [x["G"] for x in cal_frames]
        t_G, P_dimple, _ = pivot_calibration(pcd_frames)
        FT_G_frames = list()
        # get pointer markers in local coordinate frame
        g_local, _ = get_pcd_in_local_frame(cal_frames[0]["G"])
        # get all the FTs
        for frame in cal_frames:
            G_vals = frame["G"]
            # compute transform to align G with g_local
            FT_G = pcd_to_pcd_reg_w_known_correspondence(g_local, G_vals)
            FT_G_frames.append(FT_G)
        # compare FT_G_k(t_G) to P_dimple
        for FT_G in FT_G_frames:
            t_G_in_EM_tracker = FT_G.transform_pts(t_G)
            total_mse += compute_avg_mse_between_two_pcds(t_G_in_EM_tracker, P_dimple)
        avg_mse += total_mse / len(FT_G_frames)
    avg_mse = avg_mse / len(dataset_prefixes)
    # this number is somewhat arbitrary, but provides a sanity check
    assert avg_mse < 6
