from typing import Dict, List

import numpy as np

dataset_prefixes = [
    "pa2-debug-a-",
    "pa2-debug-b-",
    "pa2-debug-c-",
    "pa2-debug-d-",
    "pa2-debug-e-",
    "pa2-debug-f-",
    "pa2-unknown-g-",
    "pa2-unknown-h-",
    "pa2-unknown-i-",
    "pa2-unknown-j-",
]


def parse_calbody(path: str) -> Dict:
    """Parses a calbody.txt file according to the specifications
    in the homework description.

    Params:
        path (str): file path to the dataset file

    Returns:
        Dict: dictionary containing the d, a, and c values
    """
    assert "calbody" in path, "Wrong file."
    with open(path, "r") as file:
        data = file.readlines()
    for idx, line in enumerate(data):
        data[idx] = line.replace(",", "")
    N_D, N_A, N_C, _ = data[0].split(" ")
    N_D, N_A, N_C = int(N_D), int(N_A), int(N_C)
    idx = 1
    d = list()
    a = list()
    c = list()
    for i in range(N_D):
        d.append([float(x) for x in data[idx + i].split(" ") if x != ""])
    idx += N_D
    for i in range(N_A):
        a.append([float(x) for x in data[idx + i].split(" ") if x != ""])
    idx += N_A
    for i in range(N_C):
        c.append([float(x) for x in data[idx + i].split(" ") if x != ""])
    return {"d": np.array(d), "a": np.array(a), "c": np.array(c)}


def parse_calreadings(path: str) -> List[Dict]:
    """Parses a calreadings.txt file according to the specifications
    in the homework description.

    Params:
        path (str): file path to the dataset file

    Returns:
        List

    """
    assert "calreadings" in path, "Wrong file."
    with open(path, "r") as file:
        data = file.readlines()
    for idx, line in enumerate(data):
        data[idx] = line.replace(",", "")
    N_D, N_A, N_C, N_frames, _ = data[0].split(" ")
    N_D, N_A, N_C, N_frames = int(N_D), int(N_A), int(N_C), int(N_frames)
    idx = 1
    frames = list()
    for _ in range(N_frames):
        D = list()
        A = list()
        C = list()
        for i in range(N_D):
            D.append([float(x) for x in data[idx + i].split(" ") if x != ""])
        idx += N_D
        for i in range(N_A):
            A.append([float(x) for x in data[idx + i].split(" ") if x != ""])
        idx += N_A
        for i in range(N_C):
            C.append([float(x) for x in data[idx + i].split(" ") if x != ""])
        idx += N_C
        frames.append({"D": np.array(D), "A": np.array(A), "C": np.array(C)})
    return frames


def parse_empivot(path: str) -> List[Dict]:
    """Parses a empivot.txt file according to the specifications
    in the homework description."""
    assert "empivot" in path, "Wrong file."
    with open(path, "r") as file:
        data = file.readlines()
    for idx, line in enumerate(data):
        data[idx] = line.replace(",", "")
    N_G, N_frames, _ = data[0].split(" ")
    N_G, N_frames = int(N_G), int(N_frames)
    idx = 1
    frames = list()
    for _ in range(N_frames):
        G = list()
        for i in range(N_G):
            G.append([float(x) for x in data[idx + i].split(" ") if x != ""])
        frames.append({"G": np.array(G)})
        idx += N_G
    return frames


def parse_optpivot(path: str) -> List[Dict]:
    """Parses a empivot.txt file according to the specifications
    in the homework description."""
    assert "optpivot" in path, "Wrong file."
    with open(path, "r") as file:
        data = file.readlines()
    for idx, line in enumerate(data):
        data[idx] = line.replace(",", "")
    N_D, N_H, N_frames, _ = data[0].split(" ")
    N_D, N_H, N_frames = int(N_D), int(N_H), int(N_frames)
    idx = 1
    frames = list()
    for _ in range(N_frames):
        D = list()
        H = list()
        for i in range(N_D):
            D.append([float(x) for x in data[idx + i].split(" ") if x != ""])
        idx += N_D
        for i in range(N_H):
            H.append([float(x) for x in data[idx + i].split(" ") if x != ""])
        frames.append({"D": np.array(D), "H": np.array(H)})
        idx += N_H
    return frames


def parse_ct_fiducials(path: str) -> np.ndarray:
    """Parses a ct_fiducials.txt file according to specifications
    in the homework description."""
    assert "ct-fiducials" in path, "Wrong file path."
    with open(path, "r") as file:
        data = file.readlines()
    # get the number of fiducials
    N_B = int(data[0].split(",")[0])
    for idx, line in enumerate(data):
        data[idx] = line.replace(",", "")
    idx = 1
    b = list()
    for i in range(N_B):
        b.append([float(x) for x in data[idx + i].split(" ") if x != ""])
    return np.array(b)


def parse_em_fiducials(path: str) -> List[np.ndarray]:
    """Parses an em-fiducials.txt file according to specifications
    in the homework description."""
    assert "em-fiducials" in path, "Wrong file path."
    with open(path, "r") as file:
        data = file.readlines()
    # get the number of fiducials
    N_G, N_B = int(data[0].split(",")[0]), int(data[0].split(",")[1])
    for idx, line in enumerate(data):
        data[idx] = line.replace(",", "")
    idx = 1
    frames = list()
    for _ in range(N_B):
        G = list()
        for i in range(N_G):
            G.append([float(x) for x in data[idx + i].split(" ") if x != ""])
        idx += N_G
        frames.append(np.array(G))
    return frames


def parse_em_nav(path: str) -> List[np.ndarray]:
    """Parses an em-nav.txt file according to specifications
    in the homework description."""
    assert "em-nav" in path.lower(), "Wrong file path."
    with open(path, "r") as file:
        data = file.readlines()
    # get the number of fiducials
    N_G, N_frames = int(data[0].split(",")[0]), int(data[0].split(",")[1])
    for idx, line in enumerate(data):
        data[idx] = line.replace(",", "")
    idx = 1
    frames = list()
    for _ in range(N_frames):
        G = list()
        for i in range(N_G):
            G.append([float(x) for x in data[idx + i].split(" ") if x != ""])
        idx += N_G
        frames.append(np.array(G))
    return frames

def parse_output_2(path: str) -> List[np.ndarray]:
    """Parses an debug output file for pa2 according to specifications
    in the homework description."""
    assert "output2" in path.lower(), "Wrong file path."
    with open(path, "r") as file:
        data = file.readlines()
    pointer_points_ct = list()
    # get the number of fiducials
    for idx, line in enumerate(data[1:]):
        data[idx] = line.replace(",", "")
        temp = [float(x) for x in data[idx].split(" ") if x != ""]

        pointer_points_ct.append(np.expand_dims(np.array(temp), axis=0))
    return pointer_points_ct


if __name__ == "__main__":
    path = "/Users/byrdgb1/Desktop/Projects/CIS_1/CIS_1_PA_2/DATA/pa2-debug-a-ct-fiducials.txt"
    data = parse_ct_fiducials(path)
    path = "/Users/byrdgb1/Desktop/Projects/CIS_1/CIS_1_PA_2/DATA/pa2-debug-e-em-fiducialss.txt"
    data = parse_em_fiducials(path)
    path = (
        "/Users/byrdgb1/Desktop/Projects/CIS_1/CIS_1_PA_2/DATA/pa2-debug-e-EM-nav.txt"
    )
    data = parse_em_nav(path)
