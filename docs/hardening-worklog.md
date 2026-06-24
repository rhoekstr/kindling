# Hardening & performance worklog (autonomous run 2)

Branch: `hardening` (off master `d695628`, the merged consolidation).
Governs: [NEXT-STEPS-HARDENING-PERF.md](NEXT-STEPS-HARDENING-PERF.md).

**Run parameters (locked with the user):**
- Consolidation PR #4 **merged to master**; this run is a **new branch + new
  PR** for hardening.
- **Follow the plan order**: A1+A4 → A2/C1 → B1/B2 → A3 → B3/C2/C3 → (B4/B5/C4).
- **B2 tolerance**: ≤~0.5% relative NDCG delta OK for the latency/memory win,
  measured + documented.
- Verification gate: full pytest green + ml1m/beauty NDCG unchanged
  (0.2928 / 0.0328); steam used sparingly (~250s); **book OOMs on this 24 GB
  box — not runnable** (perf items measured on feasible proxies).
- Env: `PYTHONPATH=src .venv/bin/python`; cargo 1.95 + maturin 1.13 present.
- Delivery: commit per item; push + draft PR at the end.

---

## Log

- **Setup** — merged PR #4 → master (`d695628`); branched `hardening`.
  Pre-flight: cargo/rustc 1.95, maturin 1.13.1 available; `kindling_core` is
  a separate maturin package (`native/kindling_core/pyproject.toml`); the
  top-level hatchling build ships only the Python package (the A1 gap).
  `path/tail_index.py` still references `kindling_native` but has a clean
  Python fallback → A4 (delete v1 crate) is safe; the Rust tail fast-path
  becomes a P-perf follow-up.
