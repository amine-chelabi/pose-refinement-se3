"""SE(3) local perturbation + projection d'une droite 3D en 2D."""

from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([
        [0.0, -z,   y],
        [z,   0.0, -x],
        [-y,   x, 0.0],
    ])


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """Rodrigues : so(3) -> SO(3)."""
    theta = float(np.linalg.norm(omega))
    K = skew(omega)
    if theta < 1e-8:
        return np.eye(3) + K
    A = np.sin(theta) / theta
    B = (1.0 - np.cos(theta)) / theta ** 2
    return np.eye(3) + A * K + B * (K @ K)


def se3_increment_to_R_t(xi, R0, t0):
    """Applique xi = [omega, t_delta] à (R0, t0) — update à droite."""
    xi = np.asarray(xi, dtype=np.float64).ravel()
    omega, t_delta = xi[:3], xi[3:]
    R = R0 @ so3_exp(omega)
    t = np.asarray(t0, dtype=np.float64).ravel() + t_delta
    return R, t


def reproject_3d_line_as_2d(v, X0, t_st, t_en, K, R, t, return_endpoints=False):
    """
    Projette le segment 3D X(s) = X0 + s*v, s in [t_st, t_en], dans la caméra (K, R, t).
    Retourne la droite 2D normalisée (a, b, c), a²+b²=1, ou None si un point est derrière la caméra.
    """
    Xa_cam = R @ (X0 + t_st * v) + t
    Xb_cam = R @ (X0 + t_en * v) + t
    if Xa_cam[2] <= 1e-6 or Xb_cam[2] <= 1e-6:
        return None

    pa, pb = K @ Xa_cam, K @ Xb_cam
    xa2d = np.array([pa[0] / pa[2], pa[1] / pa[2]])
    xb2d = np.array([pb[0] / pb[2], pb[1] / pb[2]])

    l = np.cross([*xa2d, 1.0], [*xb2d, 1.0])
    norm_ab = np.linalg.norm(l[:2])
    if norm_ab < 1e-8:
        return None
    l = l / norm_ab

    return (l, xa2d, xb2d) if return_endpoints else l
