import os

import numpy as np
from PROGRAMS.main_PA1 import main
from tests.test_utils import TEST_DIR
from utils.data_processing import dataset_prefixes


def compare_output_to_debug_output(dataset_prefix: str) -> None:
    """Takes in a dataset prefix and compares the corresponding output file
    found in the OUTPUT dir with the output file found in the DATA dir.
    Raises an assertion error if the outputs do not match.
    """
    output_file_name = f"{dataset_prefix}output1.txt"
    user_output_path = os.path.join(TEST_DIR, "..", "..", "OUTPUT", output_file_name)
    debug_output_path = os.path.join(TEST_DIR, "..", "..", "DATA", output_file_name)
    with open(user_output_path, "r") as file:
        user_output = file.readlines()
    with open(debug_output_path, "r") as file:
        true_output = file.readlines()
    assert user_output[0] == true_output[0]
    total_squared_error = 0
    max_error = 0
    loc_of_max_error = None
    count = 0
    for idx in range(1, len(user_output)):
        user_data = user_output[idx].replace(",", "").replace("\n", "")
        true_data = true_output[idx].replace(",", "").replace("\n", "")
        user_data = [float(x) for x in user_data.split("\t") if x != ""]
        true_data = [float(x) for x in true_data.split(" ") if x != ""]
        user_x, user_y, user_z = user_data
        true_x, true_y, true_z = true_data
        abs_error_x = abs((user_x - true_x))
        abs_error_y = abs((user_y - true_y))
        abs_error_z = abs((user_z - true_z))
        max_local_error = max(abs_error_x, abs_error_y, abs_error_z)
        if max_local_error > max_error:
            max_error = max_local_error
            loc_of_max_error = (user_x, user_y, user_z)
        max_error = max(max_error, max_local_error)
        total_squared_error += (
            (user_x - true_x) ** 2 + (user_y - true_y) ** 2 + (user_z - true_z) ** 2
        )
        count += 3
        # assert np.isclose(user_x, true_x, 0.1)
        # assert np.isclose(user_y, true_y, 0.1)
        # assert np.isclose(user_z, true_z, 0.1)
    mse = total_squared_error / count
    assert mse < 5
    error_results_path = os.path.join(
        TEST_DIR, "..", "..", "OUTPUT", f"{dataset_prefix}results.txt"
    )
    with open(error_results_path, "w") as file:
        file.write(f"Mean squared error between points: {mse}\n")
        file.write(
            f"Maximum absolute error in point coordinate: {max_error} with coord: {loc_of_max_error}\n"
        )


def test_main_with_debug_data():
    for prefix in dataset_prefixes:
        if "debug" in prefix:
            main(prefix)
            compare_output_to_debug_output(prefix)
