import os

import numpy as np

current_script_path = os.path.abspath(__file__)
TEST_DIR = os.path.dirname(current_script_path)


def compute_avg_mse_between_two_pcds(pcd_1: np.ndarray, pcd_2: np.ndarray):
    """Computes the average mse between each pointcloud correspondence in
    a pair of pointclouds. The pointclouds are expected to be sorted so
    that the indexes in one correspond with the same index in the other."""
    assert pcd_1.shape == pcd_2.shape, "Ensure that the two pcd sets are of same len"
    num_points = pcd_1.shape[0]
    return np.sqrt(np.sum((pcd_1 - pcd_2) ** 2)) / num_points
