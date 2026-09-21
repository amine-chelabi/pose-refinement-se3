"""Test de l'optimiseur SE(3) sur une scène synthétique avec vérité terrain connue."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from se3_geometry import se3_increment_to_R_t, so3_exp, reproject_3d_line_as_2d
from pose_optimization import combined_residuals, XI_BOUNDS


def make_synthetic_scene(rng, n_lines=15, n_points=25):
    lines_3d = []
    for _ in range(n_lines):
        X0 = rng.uniform([-5, 1.5, 10], [5, 1.5, 40])
        v = rng.normal(size=3)
        v[1] = 0.0
        v /= np.linalg.norm(v)
        lines_3d.append((v, X0, 0.0, rng.uniform(0.5, 2.0)))
    pts_3d = [rng.uniform([-6, -2, 10], [6, 2, 40]) for _ in range(n_points)]
    return lines_3d, pts_3d


def project_scene(lines_3d, pts_3d, K, R, t):
    segs_obs = []
    for v, X0, t_st, t_en in lines_3d:
        _, xa2d, xb2d = reproject_3d_line_as_2d(v, X0, t_st, t_en, K, R, t, return_endpoints=True)
        segs_obs.append([*xa2d, *xb2d])

    pts_obs = []
    for X in pts_3d:
        p = K @ (R @ X + t)
        pts_obs.append([p[0] / p[2], p[1] / p[2]])

    return segs_obs, pts_obs


def pose_error(R, t, R_gt, t_gt):
    cos_theta = np.clip((np.trace(R.T @ R_gt) - 1) / 2, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_theta))
    return angle_deg, float(np.linalg.norm(t - t_gt))


def main():
    rng = np.random.default_rng(0)
    K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])
    R_gt, t_gt = np.eye(3), np.zeros(3)

    lines_3d, pts_3d = make_synthetic_scene(rng)
    segs_obs, pts_obs = project_scene(lines_3d, pts_3d, K, R_gt, t_gt)

    # dérive GPS/IMU simulée
    R_init = R_gt @ so3_exp(rng.normal(scale=np.deg2rad(1.5), size=3))
    t_init = t_gt + rng.normal(scale=0.8, size=3)

    angle_before, t_before = pose_error(R_init, t_init, R_gt, t_gt)
    print(f"avant : rot={angle_before:.2f}°  trans={t_before:.2f}m")

    result = least_squares(
        combined_residuals, np.zeros(6), bounds=XI_BOUNDS, method="trf",
        args=(lines_3d, segs_obs, pts_3d, pts_obs, K, R_init, t_init),
    )
    R_opt, t_opt = se3_increment_to_R_t(result.x, R_init, t_init)
    angle_after, t_after = pose_error(R_opt, t_opt, R_gt, t_gt)
    print(f"après : rot={angle_after:.4f}°  trans={t_after:.4f}m  "
          f"(cost={result.cost:.4f}, nfev={result.nfev}, success={result.success})")


if __name__ == "__main__":
    main()
