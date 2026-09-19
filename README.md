# Raffinement de pose caméra par contraintes géométriques PnL/PnP (SE(3))

Réimplémentation publique, à but de démonstration, d'une brique de
raffinement de pose développée durant mon stage de recherche en vision
par ordinateur chez [Imajing](https://www.imajing.io) (Toulouse), dans le
cadre d'un pipeline de correction de pose GPS smartphone par alignement
géométrique sur une caméra de référence.

> **Périmètre de ce dépôt.** Seule la brique mathématique de raffinement de
> pose (résidus, paramétrisation SE(3), optimisation) est publiée ici, sous
> forme réécrite et testée sur données synthétiques. Le pipeline complet
> (chargement des séquences, matching GlueStick, triangulation, infrastructure
> de production) reste la propriété d'Imajing et n'est pas inclus.

## Contexte et problème

Un smartphone équipé d'un GPS/IMU grand public fournit une pose (position +
orientation) bruitée, avec une dérive typique de l'ordre du mètre. On dispose
par ailleurs d'une caméra de référence, positionnée précisément, ayant
observé la même scène (route, trottoirs, marquages au sol) à un autre
instant. L'objectif est de corriger la pose du smartphone en alignant
géométriquement son image sur les structures 3D reconstruites depuis la
caméra de référence.

## Principe de la méthode

Le raffinement repose sur une optimisation non linéaire de type
Levenberg–Marquardt (`scipy.optimize.least_squares`, méthode `trf` avec
bornes), qui cherche un petit incrément de pose autour d'une estimée
initiale — d'où l'hypothèse de *warm start*.

### Paramétrisation SE(3) locale

L'incrément de pose est représenté par un vecteur de l'algèbre de Lie
$\xi = [\omega, t_\delta] \in \mathbb{R}^6$, appliqué à la pose initiale
$(R_0, t_0)$ par mise à jour à droite :

$$
R(\xi) = R_0 \cdot \exp([\omega]_\times), \qquad t(\xi) = t_0 + t_\delta
$$

où $\exp([\omega]_\times)$ est calculé par la formule de Rodrigues. Cette
paramétrisation locale à 6 degrés de liberté évite les singularités des
angles d'Euler et confine naturellement la recherche autour de la pose GPS
initiale via des bornes explicites (rotation ≤ 2.5°, translation ≤ 15 m).

### Deux types de contraintes géométriques

**PnP (Point-to-Point)** — contrainte classique : un point 3D $X$ reprojeté
doit coïncider avec son observation 2D $x_{obs}$ :

$$
r_{PnP} = \pi(K, R(\xi) X + t(\xi)) - x_{obs}
$$

**PnL (Point-to-Line)** — plus robuste sur des structures type marquage au
sol ou bord de trottoir, où l'appariement point-à-point est ambigu le long
de la ligne : on projette un segment 3D $(X_0, v, [t_{st}, t_{en}])$ en
droite 2D normalisée $l = (a, b, c)$, $a^2+b^2=1$, et on pénalise la
distance point-droite des deux extrémités observées :

$$
r_{PnL} = a \, u_{obs} + b \, v_{obs} + c
$$

Un résidu secondaire pénalise en plus le glissement du segment le long de
sa propre direction (sous-déterminé par la seule distance point-droite).

Les deux familles de résidus sont concaténées (`combined_residuals`) et
minimisées conjointement, avec un seuillage (`max_dist_px`) qui limite
l'influence des correspondances aberrantes — une forme de robustification
proche d'une perte de Huber tronquée.

## Structure du dépôt

```
se3_geometry.py     # exponentielle de Lie so(3)->SO(3), reprojection de droite 3D
pose_optimization.py# résidus PnL, PnP, combinés — cœur de l'optimisation
demo_synthetic.py   # scène 3D synthétique + optimisation + mesure d'erreur
```

## Utilisation

```bash
pip install numpy scipy
python demo_synthetic.py
```

Sortie typique (scène synthétique, dérive simulée ~1.5° / ~0.8 m) :

```
Avant optimisation : erreur rotation = 1.63°, erreur translation = 0.62 m
Après optimisation  : erreur rotation = 0.12°, erreur translation = 0.0074 m
```

## Limites

- Cette version ne couvre pas l'étape de mise en correspondance (matching de
  segments/points entre images), qui dans le pipeline original s'appuie sur
  [GlueStick](https://arxiv.org/abs/2304.02008) (Pautrat et al., ICCV 2023)
  pour l'appariement joint points + lignes, ni le filtrage RANSAC des
  correspondances en amont de l'optimisation.
- L'exemple synthétique suppose des correspondances déjà exactes (sans
  bruit de détection) ; en conditions réelles, le taux d'inliers et le bruit
  de mesure sont les facteurs limitants principaux de la précision finale.
- La paramétrisation SE(3) locale suppose une pose initiale raisonnablement
  proche de la vérité terrain (quelques degrés / quelques mètres) ; elle
  n'est pas adaptée à une estimation de pose "from scratch".

## Références

- R. Pautrat, I. Suárez, Y. Yu, M. Pollefeys, V. Larsson,
  *GlueStick: Robust Image Matching by Sticking Points and Lines Together*,
  ICCV 2023.
- J. Zaragoza, T.-J. Chin, Q.-H. Tran, M. S. Brown, D. Suter,
  *As-Projective-As-Possible Image Stitching with Moving DLT*, CVPR 2013.
