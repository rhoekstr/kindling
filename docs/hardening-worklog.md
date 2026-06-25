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

### Done & committed
- **A4 `9bfcbd1`** — deleted v1 `kindling_native` crate; path modules use
  their Python fallbacks (Rust tail fast-path = P-perf follow-up); cargo
  workspace = one crate. ml1m 0.2928, 121 tests.
- **A1 `d925ac4`** — unified maturin build: one `pip install` ships
  `kindling` + `kindling._core`. Built + installed the wheel in a clean venv
  → fit/recommend/activation_plan all work. Root `.cargo/config.toml` for the
  macOS link flags; dual-import shim keeps dev venv working; CI fixed
  (master trigger, Rust toolchain, wheels+sdist jobs); dynamic __version__.
- **A2 `d4f4c7a`** — `bench/check_gate.py` enforces the NDCG gate; new CI
  `gate` job on ml1m. ml1m PASSes; simulated 9% drop FAILs.
- **C1 `795a85e`** — mypy strict clean (26→0); CI mypy now blocking. The
  full lint gate (ruff+format+mypy) is green. ml1m 0.2928, 121 tests.

### B2 (retrieval-first latency) — MEASURED → deferred (data-grounded)
Measured recommend latency: ml1m (3.9k items) **0.8ms p50/1.9 p95**; beauty
(124k items) **6.8ms p50/9.0 p95** — sublinear, well under 10ms. The
full-catalog-scoring "cliff" only appears at 500k+ items (book), which OOMs
anyway and is gated by B1. So the retrieval-first refactor (medium-risk, and
only verifiable in the >200k regime that's memory-blocked) is **not justified
by the data**. Deferred with this rationale; spec retained in NEXT-STEPS B2.

### B1 (large-catalog memory) — root-caused; safe win shipped, real fix spec'd
- **Root cause of the book OOM**: the cooc-build peak dominates —
  `~39.8 KB/train-item × 357k ≈ 14 GB` (+ obs term) = ~17.4 GB interaction
  fit, before any extension. The cap only bounds the *extension*, not this
  floor. The real fix is a streaming/sparse cooc build (Rust) — substantial
  and unverifiable on this 24 GB box; spec'd as the B1 follow-up.
- **Bug found + fixed (verifiable)**: the cap detected RAM via `import
  psutil`, but **psutil isn't installed (not even a declared dep)** → it
  silently used an 8 GB fallback. Switched to `os.sysconf` (POSIX physical
  RAM, no dependency) so the cap actually works; added a 6 GB OS-reserve
  floor (`ceiling = min(0.80·total, total − 6 GB)`) so it fails safe
  (smaller extension / catalog-only) on constrained machines instead of
  OOMing. 8 unit tests (`test_extension_cap.py`).

### A3 / C2 / C3 / B3 — DONE
- **A3 `71a8007`** — `Engine.save/load` (versioned pickle + JSON header).
  fit→save→load recommends identically; rejects bad files/versions. 6 tests.
- **C2/C3 `9af1bb1`** — edge-case robustness already solid (empty/NaN →
  clear errors; degenerate cases degrade gracefully) → locked with tests;
  golden-output regression anchor (replaces the deleted v1↔v2 differential).
- **B3 `bf19a34`** — `bench/perf_smoke.py` (fit + serve-latency envelope).

### Final state — all of A–C landed
9 commits on `hardening`. **Suite 121 → 145; ruff+format+mypy all clean.**
Final wheel `kindling-0.2.0-cp311-abi3` installs in a clean venv and does
fit/recommend/save/load identically. Deferred (documented in NEXT-STEPS):
streaming cooc build (B1's real OOM fix), retrieval-first serving (B2),
portable persistence format, Rust path fast-path port. Pushing + opening a
new hardening PR.
