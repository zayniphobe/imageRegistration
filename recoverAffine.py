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


# B: Warp source image (from scratch)
def warp_image_fast(src_image, M, output_shape):
    """
    Warp src_image to output_shape using REVERSE MAPPING + bilinear interpolation.

    REVERSE MAPPING:
      For every pixel (x', y') in the OUTPUT, invert M to find where it came
      from in the source, then sample there with bilinear interpolation.
      This guarantees no holes in the output (unlike forward mapping).

    Bilinear interpolation (first principles):
      Given a continuous source coordinate (sx, sy), the four surrounding
      integer pixels contribute proportionally to their distance:

        I = I00*(1-dx)*(1-dy) + I10*dx*(1-dy)
          + I01*(1-dx)*dy     + I11*dx*dy

      where dx = sx - floor(sx),  dy = sy - floor(sy)
    """
    H_out, W_out = output_shape
    H_src, W_src = src_image.shape[:2]

    # Invert the affine matrix (extend to 3x3 homogeneous first)
    M_full = np.vstack([M, [0, 0, 1]])   # (3x3)
    M_inv  = np.linalg.inv(M_full)       # inverse transform

    # Build grid of every output pixel coordinate
    xs = np.arange(W_out, dtype=np.float64)
    ys = np.arange(H_out, dtype=np.float64)
    xv, yv = np.meshgrid(xs, ys)         # (H_out, W_out) each

    # Map output pixels -> source coordinates via inverse transform
    coords     = np.stack([xv.ravel(), yv.ravel(), np.ones(H_out * W_out)], axis=0)
    src_coords = M_inv @ coords           # (3, H_out*W_out)
    sx = src_coords[0].reshape(H_out, W_out)
    sy = src_coords[1].reshape(H_out, W_out)

    # Integer neighbours
    x0 = np.floor(sx).astype(int);  x1 = x0 + 1
    y0 = np.floor(sy).astype(int);  y1 = y0 + 1

    # Fractional parts (expand dim for colour channels)
    if src_image.ndim == 3:
        dx = (sx - np.floor(sx))[..., np.newaxis]
        dy = (sy - np.floor(sy))[..., np.newaxis]
    else:
        dx = sx - np.floor(sx)
        dy = sy - np.floor(sy)

    # Mask: only sample pixels whose source location is inside the image
    valid = (sx >= 0) & (sx < W_src - 1) & (sy >= 0) & (sy < H_src - 1)

    # Clamp indices so out-of-bounds accesses don't crash (masked out below)
    x0c = np.clip(x0, 0, W_src - 1);  x1c = np.clip(x1, 0, W_src - 1)
    y0c = np.clip(y0, 0, H_src - 1);  y1c = np.clip(y1, 0, H_src - 1)

    # Sample the four neighbours
    I00 = src_image[y0c, x0c].astype(np.float64)
    I10 = src_image[y0c, x1c].astype(np.float64)
    I01 = src_image[y1c, x0c].astype(np.float64)
    I11 = src_image[y1c, x1c].astype(np.float64)

    # Bilinear blend
    result = (I00 * (1 - dx) * (1 - dy) +
              I10 * dx       * (1 - dy) +
              I01 * (1 - dx) * dy       +
              I11 * dx       * dy)

    # Zero out pixels that mapped outside the source
    if src_image.ndim == 3:
        result = np.where(valid[..., np.newaxis], result, 0)
    else:
        result = np.where(valid, result, 0)

    return np.clip(result, 0, 255).astype(np.uint8)

# C: side by side comparison
def show_comparison(target_image, warped_image, title="Affine Warp Comparison", save_path=None):
    """Display target and warped images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    axes[0].imshow(target_image, cmap='gray' if target_image.ndim == 2 else None)
    axes[0].set_title("Target Image (affine_output.tif)")
    axes[0].axis('off')

    axes[1].imshow(warped_image, cmap='gray' if warped_image.ndim == 2 else None)
    axes[1].set_title("Reconstructed (Warped Source)")
    axes[1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Comparison saved -> {save_path}")
    plt.show()


# utilities
def generate_correspondences_file(M_true, image_shape, n_points=10,
                                  noise_std=0.0, filepath="correspondences.txt"):
    """
    Given a known affine matrix, generate synthetic correspondences and write
    them to a file.  noise_std > 0 simulates measurement error.
    """
    H, W = image_shape
    np.random.seed(42)

    src_x = np.random.uniform(20, W - 20, n_points)
    src_y = np.random.uniform(20, H - 20, n_points)

    ones = np.ones(n_points)
    coords     = np.stack([src_x, src_y, ones], axis=0)
    dst_coords = M_true @ coords
    dst_x = dst_coords[0].copy()
    dst_y = dst_coords[1].copy()

    if noise_std > 0:
        rng = np.random.default_rng(seed=7)
        dst_x += rng.normal(0, noise_std, n_points)
        dst_y += rng.normal(0, noise_std, n_points)

    with open(filepath, 'w') as f:
        f.write("# x  y  x'  y'   (source -> destination)\n")
        for i in range(n_points):
            f.write(f"{src_x[i]:.4f}  {src_y[i]:.4f}  {dst_x[i]:.4f}  {dst_y[i]:.4f}\n")

    print(f"  Wrote {n_points} correspondences -> '{filepath}'  (noise sigma={noise_std})")


def compute_reprojection_error(src_pts, dst_pts, M):
    """Mean & max reprojection error in pixels."""
    ones      = np.ones((len(src_pts), 1))
    src_h     = np.hstack([src_pts, ones])
    predicted = (M @ src_h.T).T
    errors    = np.linalg.norm(predicted - dst_pts, axis=1)
    print(f"  Reprojection error -- mean: {errors.mean():.4f} px,  max: {errors.max():.4f} px")
    return errors


def print_matrix_comparison(M_true, M_est):
    """Print true vs estimated matrix side by side."""
    print("  True M vs Estimated M:")
    for r in range(2):
        true_row = "  ".join(f"{v:+.6f}" for v in M_true[r])
        est_row  = "  ".join(f"{v:+.6f}" for v in M_est[r])
        print(f"    True: [{true_row}]   Est: [{est_row}]")

# runner per case
def run_case(src_image, target_image, M_true, corr_file,
             case_label, save_prefix, H, W):
    print(f"\n{'='*55}")
    print(f"  {case_label}")
    print(f"{'='*55}")

    # Part A
    src_pts, dst_pts = read_correspondences(corr_file)
    M_est = estimate_affine(src_pts, dst_pts)
    print_matrix_comparison(M_true, M_est)
    compute_reprojection_error(src_pts, dst_pts, M_est)

    # Part B
    print("  Warping image ...")
    warped = warp_image_fast(src_image, M_est, (H, W))
    Image.fromarray(warped).save(f"{save_prefix}_warped.tif")
    print(f"  Warped image saved -> '{save_prefix}_warped.tif'")

    # Part C
    show_comparison(target_image, warped,
                    title=case_label,
                    save_path=f"{save_prefix}_comparison.png")

# main
def main():
    # ── Load source image (lena)
    print("Loading lena_color_512.tif ...")
    src_image = np.array(Image.open("lena_color_512.tif").convert("RGB"))
    H, W = src_image.shape[:2]
    print(f"  Image size: {W}x{H}")

    # Define a known affine transform (rotation + scale + translation)
    # This is M_true — the transform we will try to RECOVER from correspondences
    angle  = np.deg2rad(10)   # 10 degree rotation
    scale  = 0.90             # 10% scale-down
    tx, ty = 25.0, 15.0       # translation in pixels

    M_true = np.array([
        [scale * np.cos(angle), -scale * np.sin(angle), tx],
        [scale * np.sin(angle),  scale * np.cos(angle), ty]
    ])

    print("\nTrue affine matrix M_true (rotation=10deg, scale=0.9, tx=25, ty=15):")
    print(f"  [[{M_true[0,0]:.6f}, {M_true[0,1]:.6f}, {M_true[0,2]:.6f}],")
    print(f"   [{M_true[1,0]:.6f}, {M_true[1,1]:.6f}, {M_true[1,2]:.6f}]]")

    #Generate affine_output.tif by applying M_true to lena
    print("\nGenerating affine_output.tif (applying M_true to lena) ...")
    target_image = warp_image_fast(src_image, M_true, (H, W))
    Image.fromarray(target_image).save("affine_output.tif")
    print("  Saved -> affine_output.tif")

    # Case i: Exact correspondences
    generate_correspondences_file(M_true, (H, W), n_points=10,
                                  noise_std=0.0,
                                  filepath="correspondences_exact.txt")
    run_case(src_image, target_image, M_true,
             corr_file="correspondences_exact.txt",
             case_label="Case i -- Exact Correspondences",
             save_prefix="case_exact",
             H=H, W=W)

    # Case ii: Noisy correspondences
    generate_correspondences_file(M_true, (H, W), n_points=10,
                                  noise_std=2.5,
                                  filepath="correspondences_noisy.txt")
    run_case(src_image, target_image, M_true,
             corr_file="correspondences_noisy.txt",
             case_label="Case ii -- Noisy Correspondences (sigma = 2.5 px)",
             save_prefix="case_noisy",
             H=H, W=W)

    print("\nProcess completed. Check generated files and comparison images.")


if __name__ == "__main__":
    main()