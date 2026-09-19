"""
pose_optimization.py

Fonctions de résidus pour le raffinement de pose caméra par SE(3),
combinant deux types de contraintes géométriques :

  - PnL (Point-to-Line) : distance entre les extrémités d'un segment observé
    dans l'image et la droite 2D obtenue en projetant un segment 3D de
    référence (ex. bords de trottoir, marquage au sol) sous l'hypothèse de
    pose courante.
  - PnP (Point-to-Point) : distance 2D classique entre un point 3D reprojeté
    et son observation correspondante dans l'image.

Ces résidus sont conçus pour être passés à scipy.optimize.least_squares,
qui estime l'incrément SE(3) optimal xi = [omega, t] ∈ R^6 à appliquer à une
pose initiale (R0, t0) — typiquement une pose GPS/IMU bruitée — pour la
faire coïncider avec la géométrie 3D observée dans une seconde caméra.

Contexte : réimplémentation publique, à but pédagogique et de portfolio,
d'une brique développée durant mon stage recherche chez Imajing dans le
cadre d'un pipeline de correction de pose GPS smartphone par alignement
d'images de rue multi-temporelles. Le code d'orchestration du pipeline
complet (chargement des séquences, formats propriétaires, infrastructure
de déploiement) n'est volontairement pas repris ici.
"""

from __future__ import annotations

import numpy as np

from se3_geometry import se3_increment_to_R_t, reproject_3d_line_as_2d

# Bornes de l'incrément SE(3) : rotation <= 2.5°, translation <= 15 m.
# Ces bornes traduisent l'hypothèse de "warm start" : on ne corrige qu'une
# petite dérive autour de la pose initiale, pas une pose arbitraire.
ROT_BOUND = float(np.deg2rad(2.5 / np.sqrt(3)))
T_BOUND = 15.0
XI_BOUNDS = (
    [-ROT_BOUND] * 3 + [-T_BOUND] * 3,
    [+ROT_BOUND] * 3 + [+T_BOUND] * 3,
)


def extent_residual_along_line(l_s, xa2d, xb2d, seg, max_dist_px):
    """
    Pénalise le glissement du segment projeté le long de sa propre droite
    support. Sans ce terme, la distance point-droite seule laisse le
    segment libre de glisser dans sa direction principale ; ce résidu
    secondaire (pondéré 0.25) recale les extrémités l'une par rapport à
    l'autre sans dominer la contrainte géométrique principale.
    """
    l_dir = np.array([-l_s[1], l_s[0]], dtype=np.float64)
    l_dir /= np.linalg.norm(l_dir) + 1e-8

    proj_mid = 0.5 * (float(xa2d @ l_dir) + float(xb2d @ l_dir))

    x1, y1, x2, y2 = seg
    obs_a = float(np.array([x1, y1], dtype=np.float64) @ l_dir)
    obs_b = float(np.array([x2, y2], dtype=np.float64) @ l_dir)
    obs_mid = 0.5 * (obs_a + obs_b)

    return min(abs(proj_mid - obs_mid) * 0.25, max_dist_px)


def pnl_residuals(xi, lines_3d, segs_obs, K, R0, t0, max_dist_px=50.0):
    """Résidus PnL : distance des extrémités du segment observé à la droite 3D projetée."""
    R, t = se3_increment_to_R_t(xi, R0, t0)
    t_vec = t.flatten()
    res = np.full(len(lines_3d) * 3, max_dist_px, dtype=np.float64)
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
        res[base] = min(float(np.sqrt(v1 * v1 + eps * eps) - eps), max_dist_px)
        res[base + 1] = min(float(np.sqrt(v2 * v2 + eps * eps) - eps), max_dist_px)
        res[base + 2] = extent_residual_along_line(l_s, xa2d, xb2d, seg, max_dist_px)
    return res


def pnp_residuals(xi, pts_3d, pts_obs, K, R0, t0, max_dist_px=50.0):
    """Résidus PnP : distance 2D entre point reprojeté et point observé dans l'image."""
    R, t = se3_increment_to_R_t(xi, R0, t0)
    t_vec = t.flatten()
    res = np.full(len(pts_3d) * 2, max_dist_px, dtype=np.float64)
    for i, (X, x_obs) in enumerate(zip(pts_3d, pts_obs)):
        X_cam = R @ X + t_vec
        if X_cam[2] <= 0.0:
            continue
        proj = K @ X_cam
        u = proj[0] / proj[2]
        v = proj[1] / proj[2]
        res[i * 2] = min(abs(u - float(x_obs[0])), max_dist_px)
        res[i * 2 + 1] = min(abs(v - float(x_obs[1])), max_dist_px)
    return res


def combined_residuals(xi, lines_3d, segs_obs, pts_3d, pts_obs, K, R0, t0, max_dist_px=50.0):
    """Résidus PnL + PnP concaténés pour l'optimisation SE(3) complète."""
    r_lines = pnl_residuals(xi, lines_3d, segs_obs, K, R0, t0, max_dist_px) if lines_3d else np.array([])
    r_pts = pnp_residuals(xi, pts_3d, pts_obs, K, R0, t0, max_dist_px) if pts_3d else np.array([])
    return np.concatenate([r_lines, r_pts])


def combined_pair_residuals(xi, lines_3d, segs_obs, pts_3d, pts_obs, K, R0, t0, max_dist_px=50.0):
    """Retourne un scalaire par paire (max des résidus) — utilisé pour un filtrage type RANSAC."""
    n_l = len(lines_3d)
    n_p = len(pts_3d)
    res = combined_residuals(xi, lines_3d, segs_obs, pts_3d, pts_obs, K, R0, t0, max_dist_px)
    pair_res = []
    if n_l:
        pair_res.extend(np.abs(res[:n_l * 3]).reshape(n_l, 3).max(axis=1).tolist())
    if n_p:
        pair_res.extend(np.abs(res[n_l * 3:]).reshape(n_p, 2).max(axis=1).tolist())
    return pair_res
