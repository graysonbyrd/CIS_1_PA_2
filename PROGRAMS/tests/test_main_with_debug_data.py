from main_PA2 import main
from utils.data_processing import dataset_prefixes


def test_main_with_debug_data():
    mae_list = list()
    for prefix in dataset_prefixes:
        if "debug" in prefix:
            mae, mse = main(prefix)
            mae_list.append(mae)
        assert max(mae_list) < 0.5
