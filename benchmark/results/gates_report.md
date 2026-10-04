# Benchmark Gate Report

## Gate Summary

| Gate | Value | Threshold | Pass |
|------|-------|-----------|------|
| Opening widths (≤2cm on ≥85%) | 100.0% (4/4) | ≥85% | PASS |
| Ceiling height | 0.1 cm max | ≤1.5 cm | PASS |
| Repeatability | no repeat captures | ≤1 cm or 0.5% | N/A |
| Photo stitch footprint | no photo-tier result | ≤8% | N/A |
| Calibration (90% CI coverage) | 100.0% | ~90% | PASS |

## Per-dimension errors

| Room | Dimension | Ground Truth (m) | Predicted (m) | Error (m) | Tier |
|------|-----------|------------------|---------------|-----------|------|
| room-1 | wall wall-east | 3.000 | 3.000 | -0.000 | lidar |
| room-1 | wall wall-south | 4.000 | 4.000 | +0.000 | lidar |
| room-1 | wall wall-west | 3.000 | 3.000 | -0.000 | lidar |
| room-1 | wall wall-north | 4.000 | 4.000 | +0.000 | lidar |
| room-1 | ceiling | 2.500 | 2.500 | -0.000 | lidar |
| room-1 | door door-1 | 0.860 | 0.857 | -0.003 | lidar |
| room-2 | wall wall-east | 3.000 | 3.000 | -0.000 | lidar |
| room-2 | wall wall-south | 1.200 | 1.199 | -0.001 | lidar |
| room-2 | wall wall-west | 3.000 | 3.000 | -0.000 | lidar |
| room-2 | wall wall-north | 1.200 | 1.199 | -0.001 | lidar |
| room-2 | ceiling | 2.500 | 2.499 | -0.001 | lidar |
| room-2 | door door-1 | 0.860 | 0.849 | -0.011 | lidar |
| room-2 | door door-2 | 0.860 | 0.842 | -0.018 | lidar |
| room-3 | wall wall-south | 3.500 | 3.500 | -0.000 | lidar |
| room-3 | wall wall-east | 3.000 | 2.999 | -0.001 | lidar |
| room-3 | wall wall-north | 3.500 | 3.500 | -0.000 | lidar |
| room-3 | wall wall-west | 3.000 | 2.999 | -0.001 | lidar |
| room-3 | ceiling | 2.500 | 2.499 | -0.001 | lidar |
| room-3 | door door-1 | 0.860 | 0.855 | -0.005 | lidar |

## Error statistics

- Wall length: mean 0.04 cm, max 0.08 cm (n=12)
- Ceiling height: mean 0.07 cm, max 0.11 cm (n=3)
- Opening width: mean 0.93 cm, max 1.77 cm (n=4)
- Missed openings: 0, phantom openings: 0
- Floor area: mean 0.002 m², max 0.002 m² (n=3)
