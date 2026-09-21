"""
Résidus PnL (point-to-line) + PnP (point-to-point) pour raffiner une pose
caméra par least_squares, autour d'une pose initiale (warm start GPS/IMU).
"""

from __future__ import annotations

import numpy as np

from se3_geometry import se3_increment_to_R_t, reproject_3d_line_as_2d

ROT_BOUND = float(np.deg2rad(2.5 / np.sqrt(3)))
T_BOUND = 15.0
XI_BOUNDS = (
    [-ROT_BOUND] * 3 + [-T_BOUND] * 3,
    [+ROT_BOUND] * 3 + [+T_BOUND] * 3,
)


def extent_residual_along_line(l_s, xa2d, xb2d, seg, max_dist_px):
    """Pénalise le glissement du segment le long de sa propre droite (poids 0.25, secondaire)."""
    l_dir = np.array([-l_s[1], l_s[0]])
    l_dir /= np.linalg.norm(l_dir) + 1e-8

    proj_mid = 0.5 * (xa2d @ l_dir + xb2d @ l_dir)
    x1, y1, x2, y2 = seg
    obs_mid = 0.5 * (np.array([x1, y1]) @ l_dir + np.array([x2, y2]) @ l_dir)

    return min(abs(proj_mid - obs_mid) * 0.25, max_dist_px)


def pnl_residuals(xi, lines_3d, segs_obs, K, R0, t0, max_dist_px=50.0):
    R, t = se3_increment_to_R_t(xi, R0, t0)
    t_vec = t.flatten()
    res = np.full(len(lines_3d) * 3, max_dist_px)
    for i, ((v, X0, t_st, t_en), seg) in enumerate(zip(lines_3d, segs_obs)):
        out = reproject_3d_line_as_2d(v, X0, t_st, t_en, K, R, t_vec, return_endpoints=True)
        if out is None:
            continue
        l_s, xa2d, xb2d = out
        x1, y1, x2, y2 = seg
        eps = 1e-3
        v1 = l_s[0] * x1 + l_s[1] * y1 + l_s[2]
        v2 = l_s[0] * x2 + l_s[1] * y2 + l_s[2]
        base = i * 3
        res[base] = min(np.sqrt(v1 * v1 + eps * eps) - eps, max_dist_px)
        res[base + 1] = min(np.sqrt(v2 * v2 + eps * eps) - eps, max_dist_px)
        res[base + 2] = extent_residual_along_line(l_s, xa2d, xb2d, seg, max_dist_px)
    return res


def pnp_residuals(xi, pts_3d, pts_obs, K, R0, t0, max_dist_px=50.0):
    R, t = se3_increment_to_R_t(xi, R0, t0)
    t_vec = t.flatten()
    res = np.full(len(pts_3d) * 2, max_dist_px)
    for i, (X, x_obs) in enumerate(zip(pts_3d, pts_obs)):
        X_cam = R @ X + t_vec
        if X_cam[2] <= 0.0:
            continue
        proj = K @ X_cam
        u, v = proj[0] / proj[2], proj[1] / proj[2]
        res[i * 2] = min(abs(u - x_obs[0]), max_dist_px)
        res[i * 2 + 1] = min(abs(v - x_obs[1]), max_dist_px)
    return res


def combined_residuals(xi, lines_3d, segs_obs, pts_3d, pts_obs, K, R0, t0, max_dist_px=50.0):
    r_lines = pnl_residuals(xi, lines_3d, segs_obs, K, R0, t0, max_dist_px) if lines_3d else np.array([])
    r_pts = pnp_residuals(xi, pts_3d, pts_obs, K, R0, t0, max_dist_px) if pts_3d else np.array([])
    return np.concatenate([r_lines, r_pts])


def combined_pair_residuals(xi, lines_3d, segs_obs, pts_3d, pts_obs, K, R0, t0, max_dist_px=50.0):
    """Un scalaire par paire (max des résidus) — utile pour un filtrage RANSAC en amont."""
    n_l, n_p = len(lines_3d), len(pts_3d)
    res = combined_residuals(xi, lines_3d, segs_obs, pts_3d, pts_obs, K, R0, t0, max_dist_px)
    pair_res = []
    if n_l:
        pair_res += np.abs(res[:n_l * 3]).reshape(n_l, 3).max(axis=1).tolist()
    if n_p:
        pair_res += np.abs(res[n_l * 3:]).reshape(n_p, 2).max(axis=1).tolist()
    return pair_res
