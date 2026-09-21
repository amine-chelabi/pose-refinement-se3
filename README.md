# Pose Refinement SE(3) — PnL/PnP Optimization

Camera pose refinement using point-to-point (PnP) and point-to-line (PnL)
geometric constraints, optimized via nonlinear least squares on SE(3).

Extracted and generalized from an R&D internship project at [Imajing](https://www.imajing.io)
(Toulouse) — a pipeline that corrects smartphone GPS/IMU pose by aligning
observed 3D structure (road markings, edges) against a reference camera.
Only the optimization core is published here; the full pipeline (feature
matching, triangulation, production infrastructure) is proprietary.

## Problem

A smartphone's GPS/IMU pose is noisy — typically off by ~1 meter and a few
degrees. Given a set of known 3D points/lines and their observed 2D
projections in the smartphone image, this module solves for the small pose
correction that best explains the observations.

## Method

The pose correction is parameterized as a local SE(3) increment:

```
R = R0 @ Exp(omega)      t = t0 + t_delta
```

applied around an initial pose (R0, t0), with `Exp` the so(3) exponential
map (Rodrigues' formula). Bounds are set tight (rotation ≤ 2.5°,
translation ≤ 15 m), consistent with a warm-start assumption — this refines
a locally correct pose, it does not solve pose estimation from scratch.

Two residual types feed `scipy.optimize.least_squares`:

- **PnP** — reprojection error of a 3D point against its 2D observation.
- **PnL** — distance from a reprojected 3D line to its observed segment
  endpoints. Used for line-like structure (lane markings, edges) where
  point-to-point correspondence is ambiguous along the line direction.

## Files

| File | Content |
|---|---|
| `se3_geometry.py` | so(3) exponential map, SE(3) increment, 3D line → 2D line reprojection |
| `pose_optimization.py` | PnP/PnL residuals, combined cost function |
| `demo_synthetic.py` | Validation on a synthetic scene with known ground truth |

## Usage

```bash
pip install numpy scipy
python demo_synthetic.py
```

```
avant : rot=1.63°  trans=0.62m
après : rot=0.1217°  trans=0.0074m  (cost=2.0219, nfev=14, success=True)
```

## Limitations

- Feature matching and outlier rejection (RANSAC) are not included —
  the original pipeline uses [GlueStick](https://arxiv.org/abs/2304.02008)
  (Pautrat et al., ICCV 2023) for joint point/line matching.
- Validated on synthetic data with clean correspondences; real-world
  performance depends on matching quality and inlier ratio.
- Assumes a reasonably accurate initial pose (warm start), not a
  general-purpose pose estimator.

## Author

Amine Chelabi — M2 Signal, Image et Apprentissage Automatique, Université
Toulouse III. [LinkedIn](https://www.linkedin.com/in/amine-chelabi-13726a2b7)

## License

MIT
