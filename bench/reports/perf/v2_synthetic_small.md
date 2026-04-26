# Perf report — v2 on synthetic_small

- **kindling**: 0.0.1.dev0
- **rust_core_sha**: `ad45b6d`
- **timestamp**: 2026-04-25T22:06:07
- **machine**: arm (macOS-26.5-arm64-arm-64bit-Mach-O)

## Fit

- total: **0.019s**
- peak RSS: 162 MB
- by stage:
  - `total`: 0.019s
  - `profile_decisions`: 0.050s

## Recommend

- users sampled: 200
- top-K: 10
- p50 / p95 / p99: **0.01** / 0.02 / 0.02 ms
- throughput: 64603 recs/sec
