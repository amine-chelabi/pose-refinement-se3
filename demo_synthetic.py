"""
demo_synthetic.py

Démonstration autonome (aucune donnée réelle requise) du raffinement de
pose SE(3) par résidus PnL + PnP.

Principe :
  1. On génère une scène 3D synthétique : des segments (ex. bords de route)
     et des points 3D, vus par une caméra à une pose "vraie" R_gt, t_gt.
  2. On simule une pose initiale bruitée (R_init, t_init), comme le serait
     une pose GPS/IMU smartphone avant correction.
  3. On projette la scène dans la caméra à la pose *vraie* pour obtenir des
     observations 2D "propres" (ce que verrait réellement l'image).
  4. On optimise xi (l'incrément SE(3)) avec scipy.optimize.least_squares
     pour ramener la pose initiale bruitée vers la pose vraie.
  5. On affiche l'erreur de pose avant / après optimisation.

Lancer : python demo_synthetic.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from se3_geometry import se3_increment_to_R_t, so3_exp
from pose_optimization import combined_residuals, XI_BOUNDS


def make_synthetic_scene(rng, n_lines=15, n_points=25):
    """Génère des segments 3D (route) et des points 3D (repères ponctuels)."""
    lines_3d = []
    for _ in range(n_lines):
        X0 = rng.uniform([-5, 1.5, 10], [5, 1.5, 40])   # points au sol, devant la caméra
        v = rng.normal(size=3)
        v[1] = 0.0  # segments horizontaux (type marquage au sol)
        v /= np.linalg.norm(v)
        t_st, t_en = 0.0, rng.uniform(0.5, 2.0)
        lines_3d.append((v, X0, t_st, t_en))

    pts_3d = [rng.uniform([-6, -2, 10], [6, 2, 40]) for _ in range(n_points)]
    return lines_3d, pts_3d


def project_scene(lines_3d, pts_3d, K, R, t):
    """Projette la scène avec la pose (R, t) pour obtenir les observations 2D."""
    from se3_geometry import reproject_3d_line_as_2d

    segs_obs = []
    for v, X0, t_st, t_en in lines_3d:
        out = reproject_3d_line_as_2d(v, X0, t_st, t_en, K, R, t, return_endpoints=True)
        _, xa2d, xb2d = out
        segs_obs.append([xa2d[0], xa2d[1], xb2d[0], xb2d[1]])

    pts_obs = []
    for X in pts_3d:
        X_cam = R @ X + t
        p = K @ X_cam
        pts_obs.append([p[0] / p[2], p[1] / p[2]])

    return segs_obs, pts_obs


def pose_error(R, t, R_gt, t_gt):
    """Erreur de pose : angle de rotation (deg) + distance de translation (m)."""
    R_err = R.T @ R_gt
    cos_theta = np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_theta))
    t_err = float(np.linalg.norm(t - t_gt))
    return angle_deg, t_err


def main():
    rng = np.random.default_rng(0)

    K = np.array([[1000.0, 0.0, 640.0],
                  [0.0, 1000.0, 360.0],
                  [0.0, 0.0, 1.0]])

    R_gt = np.eye(3)
    t_gt = np.array([0.0, 0.0, 0.0])

    lines_3d, pts_3d = make_synthetic_scene(rng)
    segs_obs, pts_obs = project_scene(lines_3d, pts_3d, K, R_gt, t_gt)

    # Pose initiale bruitée : simule une dérive GPS/IMU typique (~quelques
    # degrés de rotation, ~1-2 m de translation).
    omega_noise = rng.normal(scale=np.deg2rad(1.5), size=3)
    t_noise = rng.normal(scale=0.8, size=3)
    R_init = R_gt @ so3_exp(omega_noise)
    t_init = t_gt + t_noise

    angle_before, t_before = pose_error(R_init, t_init, R_gt, t_gt)
    print(f"Avant optimisation : erreur rotation = {angle_before:.2f}°, "
          f"erreur translation = {t_before:.2f} m")

    xi0 = np.zeros(6)
    result = least_squares(
        combined_residuals,
        xi0,
        bounds=XI_BOUNDS,
        args=(lines_3d, segs_obs, pts_3d, pts_obs, K, R_init, t_init),
        method="trf",
    )

    R_opt, t_opt = se3_increment_to_R_t(result.x, R_init, t_init)
    angle_after, t_after = pose_error(R_opt, t_opt, R_gt, t_gt)
    print(f"Après optimisation  : erreur rotation = {angle_after:.4f}°, "
          f"erreur translation = {t_after:.4f} m")
    print(f"Coût résiduel final : {result.cost:.4f}  "
          f"({result.nfev} évaluations, succès={result.success})")


if __name__ == "__main__":
    main()
