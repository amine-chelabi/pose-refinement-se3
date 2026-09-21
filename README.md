# Raffinement de pose caméra (PnL/PnP, SE(3))

Code extrait de mon stage R&D chez Imajing (vision par ordinateur, Toulouse).
Sert à corriger la pose GPS/IMU d'un smartphone en la recalant sur des
correspondances 3D-2D (points et segments) issues d'une caméra de référence.

Seule la brique d'optimisation est ici — matching, triangulation et
infrastructure de production restent privés.

## Fichiers

- `se3_geometry.py` : exponentielle de Lie so(3)->SO(3), reprojection d'un segment 3D en droite 2D
- `pose_optimization.py` : résidus PnL (point-to-ligne) et PnP (point-to-point), passés à `scipy.optimize.least_squares`
- `demo_synthetic.py` : test sur scène synthétique avec vérité terrain connue

## Pourquoi PnL en plus de PnP

Pour des structures type marquage au sol, l'appariement point-à-point exact
est ambigu (le point glisse le long de la ligne). Le résidu PnL pénalise la
distance à la droite reprojetée plutôt qu'à un point précis — plus robuste
sur ce type de structure.

La pose est paramétrée localement : `R = R0 @ Exp(omega)`, `t = t0 + t_delta`,
avec des bornes serrées (rotation ≤ 2.5°, translation ≤ 15 m), cohérent avec
l'hypothèse de warm start autour d'une pose GPS déjà à peu près correcte.

## Utilisation

```bash
pip install numpy scipy
python demo_synthetic.py
```

```
avant : rot=1.63°  trans=0.62m
après : rot=0.1217°  trans=0.0074m  (cost=2.0219, nfev=14, success=True)
```

## Référence

GlueStick (matching points+lignes, utilisé en amont dans le pipeline
original) : Pautrat et al., *GlueStick: Robust Image Matching by Sticking
Points and Lines Together*, ICCV 2023.
