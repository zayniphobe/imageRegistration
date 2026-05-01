import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# A: read correspondences and solve for M
def read_correspondences(filepath):
    """
    Read point correspondences from a file.
    Each line: x y x' y'
      (x, y)   — point in the SOURCE image
      (x', y') — corresponding point in the TARGET image
    Returns two arrays: src_pts, dst_pts, each shape (N, 2)
    """
    src_pts, dst_pts = [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = list(map(float, line.split()))
            src_pts.append([vals[0], vals[1]])
            dst_pts.append([vals[2], vals[3]])

    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)
    assert len(src_pts) >= 3, "Need at least 3 correspondences (6 recommended)."
    return src_pts, dst_pts