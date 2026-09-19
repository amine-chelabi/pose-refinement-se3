"""
se3_geometry.py

Utilitaires géométriques pour le raffinement de pose SE(3) :
  - paramétrisation locale d'un incrément de pose (algèbre de Lie se(3))
  - reprojection d'une droite 3D comme droite 2D dans une caméra pinhole

Ce module est une réimplémentation générique, écrite pour cette démonstration
publique, des fonctions géométriques utilisées dans le pipeline de
raffinement de pose développé durant mon stage chez Imajing. Le code
spécifique à l'infrastructure de production (chargement des séquences,
formats propriétaires, etc.) n'est pas repris ici : seule la partie
mathématique, générique et réutilisable, est publiée.

Convention SE(3) : un incrément xi = [omega (3,), t_delta (3,)] est appliqué
autour d'une pose de référence (R0, t0) par mise à jour à droite :

    R(xi) = R0 @ Exp(omega)
    t(xi) = t0 + t_delta

où Exp(omega) est l'exponentielle de Lie de so(3), calculée via la formule
de Rodrigues. Cette paramétrisation locale est adaptée à un raffinement
"warm start" : on optimise un petit déplacement autour d'une pose initiale
(typiquement issue du GPS/IMU du smartphone) plutôt que la pose absolue.
"""

from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """Matrice antisymétrique associée à v ∈ R^3, telle que skew(v) @ x = v × x."""
    x, y, z = v
    return np.array([
        [0.0, -z,   y],
        [z,   0.0, -x],
        [-y,   x, 0.0],
    ], dtype=np.float64)


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """
    Exponentielle de Lie so(3) -> SO(3) via la formule de Rodrigues.

    R = I + sin(theta)/theta * [omega]_x + (1 - cos(theta))/theta^2 * [omega]_x^2

    avec theta = ||omega||. Le cas theta -> 0 est régularisé par un
    développement limité pour éviter la division par zéro.
    """
    theta = float(np.linalg.norm(omega))
    K = skew(omega)
    if theta < 1e-8:
        # Développement limité au 1er ordre : R ≈ I + [omega]_x
        return np.eye(3) + K
    A = np.sin(theta) / theta
    B = (1.0 - np.cos(theta)) / (theta ** 2)
    return np.eye(3) + A * K + B * (K @ K)


def se3_increment_to_R_t(
    xi: np.ndarray,
    R0: np.ndarray,
    t0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applique un incrément SE(3) local xi = [omega, t_delta] à une pose de
    référence (R0, t0).

    Args:
        xi : vecteur (6,) = [omega_x, omega_y, omega_z, tx, ty, tz]
        R0 : rotation initiale (3, 3)
        t0 : translation initiale (3,)

    Returns:
        (R, t) : pose mise à jour
    """
    xi = np.asarray(xi, dtype=np.float64).ravel()
    omega, t_delta = xi[:3], xi[3:]
    R = R0 @ so3_exp(omega)
    t = np.asarray(t0, dtype=np.float64).ravel() + t_delta
    return R, t


def reproject_3d_line_as_2d(
    v: np.ndarray,
    X0: np.ndarray,
    t_st: float,
    t_en: float,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    return_endpoints: bool = False,
):
    """
    Projette un segment 3D, défini par une droite paramétrique
    X(s) = X0 + s * v pour s ∈ [t_st, t_en], dans une caméra pinhole (K, R, t),
    et retourne la droite 2D résultante sous forme normalisée (a, b, c)
    telle que a*u + b*v + c = 0 avec a^2 + b^2 = 1 (distance point-droite
    directement lisible comme |a*u + b*v + c|).

    Args:
        v, X0, t_st, t_en : paramétrisation du segment 3D
        K                 : matrice intrinsèque (3, 3)
        R, t              : pose caméra courante (monde -> caméra)
        return_endpoints  : si True, retourne aussi les extrémités 2D projetées

    Returns:
        None si un des deux points est derrière la caméra (profondeur <= 0),
        sinon (l, xa2d, xb2d) si return_endpoints, ou l seul.
    """
    Xa = X0 + t_st * v
    Xb = X0 + t_en * v

    Xa_cam = R @ Xa + t
    Xb_cam = R @ Xb + t
    if Xa_cam[2] <= 1e-6 or Xb_cam[2] <= 1e-6:
        return None

    pa = K @ Xa_cam
    pb = K @ Xb_cam
    xa2d = np.array([pa[0] / pa[2], pa[1] / pa[2]])
    xb2d = np.array([pb[0] / pb[2], pb[1] / pb[2]])

    # Droite 2D passant par xa2d, xb2d (produit vectoriel en coordonnées homogènes)
    pa_h = np.array([xa2d[0], xa2d[1], 1.0])
    pb_h = np.array([xb2d[0], xb2d[1], 1.0])
    l = np.cross(pa_h, pb_h)

    norm_ab = np.linalg.norm(l[:2])
    if norm_ab < 1e-8:
        return None
    l = l / norm_ab  # normalisation a^2 + b^2 = 1

    if return_endpoints:
        return l, xa2d, xb2d
    return l
