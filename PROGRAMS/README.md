# Grayson Byrd and Akhil Deo CIS I PA 2

## Installation
Navigate to the PROGRAMS/ directory and run the following to create the required programming environment to run the code.

```bash
conda create -n cis python=3.11
conda activate cis
pip install -r requirements.txt
```

## Testing
To run all tests, run the following command:

```bash
pytest
```

This should show all 7 tests are passing.

## Full Run

All datasets should already by located in the DATA folder. Additionally, all problem outputs should already be located in the OUTPUT folder. To re-run my final main script on each dataset, simply navigate to the PROGRAMS/ directory and run the following:

```bash
python main_PA2.py --full_run
```

This will overwrite any files that are already present in the OUTPUT folder.

## Partial Run

To run the main script on a single provided dataset, run:

```bash
python main.py --dataset_prefix <dataset_prefix>
```

Where dataset prefix is the prefix to one of the datasets in the DATA folder (e.g. pa2-debug-a- is a prefix).
