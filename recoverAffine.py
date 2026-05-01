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

def estimate_affine(src_pts, dst_pts):
    """
    Solve for the 2x3 affine matrix M that maps src -> dst.

    For each correspondence (x,y) -> (x', y'):
        x' = a*x + b*y + tx      (row 1 of M)
        y' = c*x + d*y + ty      (row 2 of M)

    Stacking N pairs gives the overdetermined system  A * p = b:
        A  is (2N x 6)
        p  = [a, b, tx, c, d, ty]^T
        b  = [x1', y1', x2', y2', ...]^T

    np.linalg.lstsq finds the least-squares solution automatically.

    Returns M -- shape (2, 3):
        [[a,  b,  tx],
         [c,  d,  ty]]
    """
    N = len(src_pts)
    A = np.zeros((2 * N, 6))
    b = np.zeros(2 * N)

    for i, ((x, y), (xp, yp)) in enumerate(zip(src_pts, dst_pts)):
        A[2*i,   :] = [x, y, 1,  0, 0, 0]
        b[2*i]      = xp
        A[2*i+1, :] = [0, 0, 0,  x, y, 1]
        b[2*i+1]    = yp

    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, bx, tx, c, d, ty = params

    M = np.array([[a,  bx, tx],
                  [c,  d,  ty]])

    print("  Estimated affine matrix M:")
    print(f"    [[{a:.6f}, {bx:.6f}, {tx:.6f}],")
    print(f"     [{c:.6f}, {d:.6f}, {ty:.6f}]]")
    return M