from pathlib import Path
import csv
import numpy as np

# Desired ordering of the basenames (no “.csv”)
_ORDER = [
    "HH", "HV", "VV", "VH",
    "RH", "RV", "DV", "DH",
    "DR", "DD", "RD", "HD",
    "VD", "VL", "HL", "RL",
]


def load_csv_results(csv_path):
    """
    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to the CSV file with two columns:
        - filename (e.g. “HH.csv”)
        - numeric value

    Returns
    -------
    np.ndarray
        1-D array with 16 values arranged in the order given by _ORDER.
    """
    csv_path = Path(csv_path)

    # Read the file and build a lookup {basename -> value}
    lookup = {}
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)              # skip header
        for filename, value in reader:
            basename = filename.removesuffix(".csv")
            lookup[basename] = float(value)

    # Assemble the array in the exact order
    try:
        return np.array([lookup[key] for key in _ORDER], dtype=float)
    except KeyError as missing:
        raise ValueError(f"Missing entry for {missing.args[0]!r} in {csv_path}") from None
        
"""
from csv_tomography import load_csv_results

counts = load_csv_results("measurements.csv")
# counts is now a NumPy array with 16 elements in the required order

"""
