//! Native engine state + recommend — a Rust object that owns the fit arrays and
//! reproduces `engine.py::_recommend_core` end to end.
//!
//! Parity-first: the state is *built from* the Python-fitted `EngineState`
//! (Python keeps ingest / preprocess / fit orchestration for now), then the
//! native `recommend` runs over the owned arrays — no per-call marshaling of
//! the dense EASE matrix. This pass covers the EASE base + channel blend +
//! temporal-cooc boost layer (ml1m / beauty / steam all use base = ease).
//! cold-slots (steam) and the cooc base (book) land next.

use ndarray::ArrayView1;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rayon::prelude::*;

use crate::engine::recommend::{blend_full, BlendState};
use crate::repeat::multiplier::multiplier_one;
use crate::score::layered::{layered_score, ZMode};

type Csr32 = (Vec<f32>, Vec<i32>, Vec<i32>);

/// Native recommend engine. Constructed via [`build_engine`] from a Python
/// `EngineState`; `recommend` reproduces `_recommend_core`. Serializable
/// (bincode) so a fitted engine can be persisted as a self-contained serving
/// artifact and reloaded without re-fitting.
#[pyclass]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct EngineState {
    n_items: usize,
    // base: EASE (dense ease_n×ease_n) or cooc-fused (symmetric cooc CSR).
    base_is_ease: bool,
    ease_n: usize,
    ease_b: Vec<f32>, // ease_n × ease_n, row-major
    cooc_data: Vec<f32>,
    cooc_indices: Vec<i32>,
    cooc_indptr: Vec<i32>,

    // channels
    trend_z: Vec<f64>,
    trend_alpha: f64,
    last_item_alpha: f64,
    trans_data: Vec<f64>,
    trans_indices: Vec<i32>,
    trans_indptr: Vec<i64>,
    transition_alpha: f64,
    transition_last_k: usize,
    transition_decay: f64,
    uu_data: Vec<i64>,
    uu_indptr: Vec<i64>,
    uu_deg: Vec<f64>,
    uri_data: Vec<i64>,
    uri_indptr: Vec<i64>,
    user_cf_alpha: f64,
    user_cf_k: usize,
    n_users: usize,

    // popularity prior (new-user path; recommend passes pop_prior=0)
    item_pop: Vec<f64>,

    // boost layers (temporal_cooc / session_cooc): symmetric CSR adjacencies
    boost: Vec<Csr32>,
    z_threshold: f64,
    boost_multiplier: f64,
    retrieval_budget: usize,

    // cold-slots: content-space ranker over the "new releases shelf"
    cold_slots: usize,
    content_data: Vec<f32>,
    content_indices: Vec<i32>,
    content_indptr: Vec<i32>,
    content_nfeat: usize,
    content_coldness: Vec<f64>,
    cold_recency: Vec<f64>,
    cold_recency_beta: f64,
    content_alpha: f64,

    // repeat module: per-user CSR of reorder items re-surfaced via the timing
    // multiplier (REPLENISH). Active only on repeat-regime datasets.
    repeat_active: bool,
    repeat_indptr: Vec<i64>,
    repeat_items: Vec<i64>,
    repeat_last_ts: Vec<f64>,
    repeat_periods: Vec<f64>,
    repeat_quality: Vec<f64>,
    repeat_now_ts: f64,
    repeat_refractory: f64,
    repeat_epsilon: f64,
}

fn f64v(d: &Bound<'_, PyDict>, k: &str) -> PyResult<Vec<f64>> {
    match d.get_item(k)? {
        Some(o) if !o.is_none() => Ok(o.extract::<PyReadonlyArray1<'_, f64>>()?.as_slice()?.to_vec()),
        _ => Ok(Vec::new()),
    }
}
fn i64v(d: &Bound<'_, PyDict>, k: &str) -> PyResult<Vec<i64>> {
    match d.get_item(k)? {
        Some(o) if !o.is_none() => Ok(o.extract::<PyReadonlyArray1<'_, i64>>()?.as_slice()?.to_vec()),
        _ => Ok(Vec::new()),
    }
}
fn i32v(d: &Bound<'_, PyDict>, k: &str) -> PyResult<Vec<i32>> {
    match d.get_item(k)? {
        Some(o) if !o.is_none() => Ok(o.extract::<PyReadonlyArray1<'_, i32>>()?.as_slice()?.to_vec()),
        _ => Ok(Vec::new()),
    }
}
fn f32v(d: &Bound<'_, PyDict>, k: &str) -> PyResult<Vec<f32>> {
    match d.get_item(k)? {
        Some(o) if !o.is_none() => Ok(o.extract::<PyReadonlyArray1<'_, f32>>()?.as_slice()?.to_vec()),
        _ => Ok(Vec::new()),
    }
}
fn cfg_f64(d: &Bound<'_, PyDict>, k: &str, default: f64) -> PyResult<f64> {
    match d.get_item(k)? {
        Some(o) if !o.is_none() => o.extract::<f64>(),
        _ => Ok(default),
    }
}
fn cfg_usize(d: &Bound<'_, PyDict>, k: &str, default: usize) -> PyResult<usize> {
    match d.get_item(k)? {
        Some(o) if !o.is_none() => o.extract::<usize>(),
        _ => Ok(default),
    }
}
fn cfg_bool(d: &Bound<'_, PyDict>, k: &str, default: bool) -> PyResult<bool> {
    match d.get_item(k)? {
        Some(o) if !o.is_none() => o.extract::<bool>(),
        _ => Ok(default),
    }
}

/// Build a native [`EngineState`] from the Python `EngineState` arrays
/// (`arrays`) and scalar config (`config`). Arrays absent / None become empty
/// (channel off). `ease_b` is a 2-D float32 array; `boost` is a list of
/// `(data, indices, indptr)` CSR tuples.
#[pyfunction]
fn build_engine(arrays: &Bound<'_, PyDict>, config: &Bound<'_, PyDict>) -> PyResult<EngineState> {
    let (ease_b, ease_n) = match arrays.get_item("ease_b")? {
        Some(o) if !o.is_none() => {
            let a = o.extract::<PyReadonlyArray2<'_, f32>>()?;
            let v = a.as_array();
            (v.iter().copied().collect::<Vec<f32>>(), v.ncols())
        }
        _ => (Vec::new(), 0),
    };
    // Shape validation: a base must exist, fit the catalog, and be square.
    let n_items = cfg_usize(config, "n_items", 0)?;
    let has_cooc = !i32v(arrays, "cooc_indptr")?.is_empty();
    if n_items == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("n_items must be > 0"));
    }
    if ease_n == 0 && !has_cooc {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "no base scorer: provide ease_b or a cooc CSR",
        ));
    }
    if ease_n > n_items || ease_b.len() != ease_n * ease_n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "ease_b shape inconsistent: dim {ease_n}, len {}, n_items {n_items}",
            ease_b.len()
        )));
    }
    let mut boost: Vec<Csr32> = Vec::new();
    if let Some(o) = arrays.get_item("boost")? {
        if !o.is_none() {
            for spec in o.downcast::<PyList>()?.iter() {
                let t = spec.downcast::<PyTuple>()?;
                let data = t.get_item(0)?.extract::<PyReadonlyArray1<'_, f32>>()?.as_slice()?.to_vec();
                let indices = t.get_item(1)?.extract::<PyReadonlyArray1<'_, i32>>()?.as_slice()?.to_vec();
                let indptr = t.get_item(2)?.extract::<PyReadonlyArray1<'_, i32>>()?.as_slice()?.to_vec();
                boost.push((data, indices, indptr));
            }
        }
    }
    Ok(EngineState {
        n_items,
        base_is_ease: cfg_bool(config, "base_is_ease", true)?,
        ease_n,
        ease_b,
        cooc_data: f32v(arrays, "cooc_data")?,
        cooc_indices: i32v(arrays, "cooc_indices")?,
        cooc_indptr: i32v(arrays, "cooc_indptr")?,
        trend_z: f64v(arrays, "trend_z")?,
        trend_alpha: cfg_f64(config, "trend_alpha", 0.0)?,
        last_item_alpha: cfg_f64(config, "last_item_alpha", 0.0)?,
        trans_data: f64v(arrays, "trans_data")?,
        trans_indices: i32v(arrays, "trans_indices")?,
        trans_indptr: i64v(arrays, "trans_indptr")?,
        transition_alpha: cfg_f64(config, "transition_alpha", 0.0)?,
        transition_last_k: cfg_usize(config, "transition_last_k", 5)?,
        transition_decay: cfg_f64(config, "transition_decay", 0.7)?,
        uu_data: i64v(arrays, "uu_data")?,
        uu_indptr: i64v(arrays, "uu_indptr")?,
        uu_deg: f64v(arrays, "uu_deg")?,
        uri_data: i64v(arrays, "uri_data")?,
        uri_indptr: i64v(arrays, "uri_indptr")?,
        user_cf_alpha: cfg_f64(config, "user_cf_alpha", 0.0)?,
        user_cf_k: cfg_usize(config, "user_cf_k", 100)?,
        n_users: cfg_usize(config, "n_users", 0)?,
        item_pop: f64v(arrays, "item_pop")?,
        boost,
        z_threshold: cfg_f64(config, "z_threshold", 2.5)?,
        boost_multiplier: cfg_f64(config, "boost_multiplier", 3.0)?,
        retrieval_budget: cfg_usize(config, "retrieval_budget", 500)?,
        cold_slots: cfg_usize(config, "cold_slots", 0)?,
        content_data: f32v(arrays, "content_data")?,
        content_indices: i32v(arrays, "content_indices")?,
        content_indptr: i32v(arrays, "content_indptr")?,
        content_nfeat: cfg_usize(config, "content_nfeat", 0)?,
        content_coldness: f64v(arrays, "content_coldness")?,
        cold_recency: f64v(arrays, "cold_recency")?,
        cold_recency_beta: cfg_f64(config, "cold_recency_beta", 0.0)?,
        content_alpha: cfg_f64(config, "content_alpha", 0.0)?,
        repeat_active: cfg_bool(config, "repeat_active", false)?,
        repeat_indptr: i64v(arrays, "repeat_indptr")?,
        repeat_items: i64v(arrays, "repeat_items")?,
        repeat_last_ts: f64v(arrays, "repeat_last_ts")?,
        repeat_periods: f64v(arrays, "repeat_periods")?,
        repeat_quality: f64v(arrays, "repeat_quality")?,
        repeat_now_ts: cfg_f64(config, "repeat_now_ts", f64::NAN)?,
        repeat_refractory: cfg_f64(config, "repeat_refractory", 0.0)?,
        repeat_epsilon: cfg_f64(config, "repeat_epsilon", 1e-3)?,
    })
}

/// Top-`budget` finite indices of `s`, sorted by score desc then index asc
/// (== Python `argpartition` + stable `argsort`, with owned masked to -inf so
/// non-finite entries drop out).
fn top_indices(s: &[f64], budget: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..s.len()).filter(|&i| s[i].is_finite()).collect();
    let k = budget.min(idx.len());
    if k == 0 {
        return Vec::new();
    }
    let cmp = |&a: &usize, &c: &usize| {
        s[c].partial_cmp(&s[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&c))
    };
    if idx.len() > k {
        idx.select_nth_unstable_by(k - 1, cmp);
        idx.truncate(k);
    }
    idx.sort_by(cmp);
    idx
}

#[pymethods]
impl EngineState {
    /// Override the channel blend weights in place — used by the held-out
    /// channel-activation gate to ablate a channel without rebuilding the
    /// (dense-EASE-carrying) state.
    fn set_channel_alphas(
        &mut self,
        trend_alpha: f64,
        user_cf_alpha: f64,
        last_item_alpha: f64,
        transition_alpha: f64,
    ) {
        self.trend_alpha = trend_alpha;
        self.user_cf_alpha = user_cf_alpha;
        self.last_item_alpha = last_item_alpha;
        self.transition_alpha = transition_alpha;
    }

    /// Top-n recommendations for `owned` (engine item indices). `user_row` is
    /// the entity's user-CF row (-1 if unknown); `pop_prior` drives the
    /// new-user popularity addend (0 for known users). Returns
    /// `(item_indices, scores, base_kind)`.
    #[pyo3(signature = (owned, user_row, n, pop_prior=0.0))]
    fn recommend(
        &self,
        owned: Vec<i64>,
        user_row: i64,
        n: usize,
        pop_prior: f64,
    ) -> (Vec<i64>, Vec<f64>, Vec<String>) {
        self.recommend_one(&owned, user_row, n, pop_prior)
    }

    /// Batch recommend over many users in parallel, with the GIL released
    /// (rayon). `owneds[i]` / `user_rows[i]` describe user i; equivalent to
    /// calling `recommend` per user. This is the full-catalog eval win — no
    /// GIL, so the per-user EASE sums / blends / retrievals run concurrently.
    #[pyo3(signature = (owneds, user_rows, n, pop_prior=0.0))]
    fn recommend_batch(
        &self,
        py: Python<'_>,
        owneds: Vec<Vec<i64>>,
        user_rows: Vec<i64>,
        n: usize,
        pop_prior: f64,
    ) -> Vec<(Vec<i64>, Vec<f64>, Vec<String>)> {
        py.allow_threads(|| {
            owneds
                .par_iter()
                .zip(user_rows.par_iter())
                .map(|(ow, &ur)| self.recommend_one(ow, ur, n, pop_prior))
                .collect()
        })
    }

    /// Serialize the native engine to `path` (bincode) — a self-contained
    /// serving artifact that reloads with [`EngineState::load`] (no re-fit).
    fn save(&self, path: &str) -> PyResult<()> {
        let bytes = bincode::serialize(self)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        std::fs::write(path, bytes)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Load a native engine saved with [`EngineState::save`].
    #[staticmethod]
    fn load(path: &str) -> PyResult<EngineState> {
        let bytes = std::fs::read(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        bincode::deserialize(&bytes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Serialize to an in-memory bincode buffer.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let bytes = bincode::serialize(self)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(pyo3::types::PyBytes::new_bound(py, &bytes))
    }

    /// Reconstruct from a [`EngineState::to_bytes`] buffer.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<EngineState> {
        bincode::deserialize(data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

impl EngineState {
    /// Core recommend (no PyO3) — shared by `recommend` and `recommend_batch`.
    fn recommend_one(
        &self,
        owned: &[i64],
        user_row: i64,
        n: usize,
        pop_prior: f64,
    ) -> (Vec<i64>, Vec<f64>, Vec<String>) {
        let n_items = self.n_items;
        // Defensive: the PyO3 entry point is callable with arbitrary indices.
        // Drop negative / out-of-range owned items so no downstream index (the
        // owned mask, the EASE/cooc/content row gathers) can panic.
        let owned: Vec<i64> = owned
            .iter()
            .copied()
            .filter(|&o| o >= 0 && (o as usize) < n_items)
            .collect();
        let owned = owned.as_slice();
        // No (valid) history → no personalized recommendation. The serving
        // layer routes zero-history users to a popularity fallback separately.
        if owned.is_empty() {
            return (Vec::new(), Vec::new(), Vec::new());
        }
        // Plain cooc (cooc base with no *active* trend/transition channel) skips
        // the blend entirely and applies a positivity filter — matches the
        // Python `_recommend_core` else-branch (which gates on the channel
        // arrays being present, not just alpha > 0). Otherwise: EASE or
        // cooc-fused (blended).
        let trend_active = self.trend_alpha > 0.0 && self.trend_z.len() == n_items;
        let trans_active = self.transition_alpha > 0.0 && !self.trans_indptr.is_empty();
        let plain_cooc = !self.base_is_ease && !(trend_active || trans_active);
        let base_kind = if self.base_is_ease {
            "ease"
        } else if plain_cooc {
            "cooc"
        } else {
            "cooc_fused"
        };
        // base_vec: EASE row-sum (padded to n_items) or cooc-fused row-sum.
        let mut base = vec![0f64; n_items];
        if self.base_is_ease {
            for &o in owned {
                let o = o as usize;
                if o < self.ease_n {
                    let row = &self.ease_b[o * self.ease_n..(o + 1) * self.ease_n];
                    for (b, &x) in base.iter_mut().zip(row) {
                        *b += x as f64;
                    }
                }
            }
        } else {
            for &o in owned {
                let o = o as usize;
                if o + 1 < self.cooc_indptr.len() {
                    for k in self.cooc_indptr[o] as usize..self.cooc_indptr[o + 1] as usize {
                        base[self.cooc_indices[k] as usize] += self.cooc_data[k] as f64;
                    }
                }
            }
        }
        // last-item row (padded) — EASE base only (no EASE row for cooc base).
        let mut last_row: Vec<f64> = Vec::new();
        if self.base_is_ease && self.last_item_alpha > 0.0 && !owned.is_empty() {
            let last = *owned.last().unwrap() as usize;
            last_row = vec![0f64; n_items];
            if last < self.ease_n {
                for (d, &x) in last_row
                    .iter_mut()
                    .zip(&self.ease_b[last * self.ease_n..(last + 1) * self.ease_n])
                {
                    *d = x as f64;
                }
            }
        }
        // Content channel: coldness · z(content_scores), added as
        // content_alpha · contrib. Skipped for plain cooc (no blend).
        let content_contrib: Vec<f64> = if self.content_alpha > 0.0
            && self.content_nfeat > 0
            && !plain_cooc
        {
            let cs = self.content_scores(owned);
            let nn = cs.len() as f64;
            let mean = cs.iter().sum::<f64>() / nn;
            let std = (cs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / nn).sqrt();
            if std > 0.0 {
                let has_cold = self.content_coldness.len() == n_items;
                cs.iter()
                    .enumerate()
                    .map(|(j, &x)| {
                        let cold = if has_cold { self.content_coldness[j] } else { 1.0 };
                        cold * (x - mean) / std
                    })
                    .collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        let mut scores = if plain_cooc {
            base
        } else {
            let bs = BlendState {
                trend_z: &self.trend_z,
                trend_alpha: self.trend_alpha,
                last_row: &last_row,
                last_item_alpha: self.last_item_alpha,
                trans_data: &self.trans_data,
                trans_indices: &self.trans_indices,
                trans_indptr: &self.trans_indptr,
                transition_alpha: self.transition_alpha,
                transition_last_k: self.transition_last_k,
                transition_decay: self.transition_decay,
                uu_data: &self.uu_data,
                uu_indptr: &self.uu_indptr,
                uu_deg: &self.uu_deg,
                uri_data: &self.uri_data,
                uri_indptr: &self.uri_indptr,
                user_cf_alpha: self.user_cf_alpha,
                user_cf_k: self.user_cf_k,
                user_row,
                n_users: self.n_users,
                content_alpha: self.content_alpha,
                content_contrib: &content_contrib,
            };
            blend_full(base, n_items, owned, &bs)
        };
        // New-user popularity addend: pop_prior · z(log1p(pop)).
        if pop_prior > 0.0 && self.item_pop.len() == n_items {
            let p: Vec<f64> = self.item_pop.iter().map(|&x| (x + 1.0).ln()).collect();
            let nn = n_items as f64;
            let mean = p.iter().sum::<f64>() / nn;
            let sd = (p.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / nn).sqrt();
            if sd > 0.0 {
                for (s, &pi) in scores.iter_mut().zip(&p) {
                    *s += pop_prior * (pi - mean) / sd;
                }
            }
        }
        // Repeat regime: exempt the user's reorder items from the owned-mask,
        // re-surfacing them at base × REPLENISH-multiplier (suppress just-bought,
        // keep due). Computed from the pre-mask blended scores. Timestamp-less
        // logs → period NaN → multiplier 1.0 (surface on base affinity alone).
        let mut exempt: Vec<(usize, f64)> = Vec::new();
        if self.repeat_active
            && user_row >= 0
            && (user_row as usize) + 1 < self.repeat_indptr.len()
        {
            let ur = user_row as usize;
            for k in self.repeat_indptr[ur] as usize..self.repeat_indptr[ur + 1] as usize {
                let j = self.repeat_items[k] as usize;
                if j >= n_items {
                    continue;
                }
                let tsl = if self.repeat_last_ts[k].is_finite() {
                    self.repeat_now_ts - self.repeat_last_ts[k]
                } else {
                    f64::NAN
                };
                let mult = multiplier_one(
                    [0.0, 1.0, 0.0, 0.0], // REPLENISH
                    self.repeat_periods[k],
                    self.repeat_refractory,
                    self.repeat_quality[k],
                    tsl,
                    self.repeat_epsilon,
                );
                exempt.push((j, scores[j] * mult));
            }
        }
        // Exclude owned.
        for &o in owned {
            scores[o as usize] = f64::NEG_INFINITY;
        }
        // Re-surface repeat-profile items (override the mask).
        for (j, v) in exempt {
            scores[j] = v;
        }
        // Retrieve candidate pool.
        let cand = top_indices(&scores, self.retrieval_budget.min(n_items));
        if cand.is_empty() {
            return (Vec::new(), Vec::new(), Vec::new());
        }
        // Base scores over the pool (EASE path carries its own pool scores).
        let base_cand: Vec<f64> = cand.iter().map(|&c| scores[c]).collect();
        // Boost layers: cooc-signal over the candidate pool, sparse z-mode.
        let mut layers: Vec<(Vec<f64>, ZMode)> = Vec::with_capacity(self.boost.len());
        for (data, indices, indptr) in &self.boost {
            let nrow = indptr.len().saturating_sub(1);
            let mut summed = vec![0f64; nrow];
            for &o in owned {
                let o = o as usize;
                if o + 1 < indptr.len() {
                    for k in indptr[o] as usize..indptr[o + 1] as usize {
                        summed[indices[k] as usize] += data[k] as f64;
                    }
                }
            }
            let ls: Vec<f64> = cand
                .iter()
                .map(|&c| if c < summed.len() { summed[c] } else { 0.0 })
                .collect();
            layers.push((ls, ZMode::NonzeroSubset));
        }
        let composite = layered_score(
            ArrayView1::from(&base_cand),
            &layers,
            self.z_threshold,
            self.boost_multiplier,
            20,
            3,
        );
        // Top-n by composite desc (ties by pool position asc). EASE base ⇒ no
        // positivity filter (signed weights).
        let mut order: Vec<usize> = (0..cand.len()).collect();
        let k = n.min(order.len());
        let cmp = |&a: &usize, &c: &usize| {
            composite[c].partial_cmp(&composite[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&c))
        };
        if order.len() > k {
            if k > 0 {
                order.select_nth_unstable_by(k - 1, cmp);
            }
            order.truncate(k);
        }
        order.sort_by(cmp);
        // Plain cooc: positivity filter — a 0 composite means no co-occurrence
        // evidence (EASE / cooc-fused weights are signed, so no filter there).
        if base_kind == "cooc" {
            order.retain(|&i| composite[i] > 0.0);
        }
        let mut items: Vec<i64> = order.iter().map(|&i| cand[i] as i64).collect();
        let mut out_scores: Vec<f64> = order.iter().map(|&i| composite[i]).collect();
        let mut kinds: Vec<String> = vec![base_kind.to_string(); items.len()];

        // Reserved cold slots ("new releases shelf"): content-space ranker over
        // the cold tail, replacing the final `cold_slots` warm picks.
        if self.cold_slots > 0 && self.content_nfeat > 0 && self.content_coldness.len() == n_items {
            let mut cs = self.content_scores(owned);
            let nn = cs.len() as f64;
            let mean = cs.iter().sum::<f64>() / nn;
            let std = (cs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / nn).sqrt();
            if std > 0.0 {
                for x in cs.iter_mut() {
                    *x = (*x - mean) / std;
                }
            }
            if self.cold_recency.len() == n_items && self.cold_recency_beta > 0.0 {
                for (x, &r) in cs.iter_mut().zip(&self.cold_recency) {
                    *x += self.cold_recency_beta * r;
                }
            }
            for j in 0..n_items {
                if self.content_coldness[j] < 0.75 {
                    cs[j] = f64::NEG_INFINITY;
                }
            }
            for &o in owned {
                cs[o as usize] = f64::NEG_INFINITY;
            }
            let kept: std::collections::HashSet<i64> = items.iter().copied().collect();
            let mut cold_cand: Vec<usize> = (0..n_items).filter(|&i| cs[i].is_finite()).collect();
            cold_cand.sort_by(|&a, &b| {
                cs[b].partial_cmp(&cs[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
            });
            let mut cold_picks: Vec<usize> = Vec::new();
            for i in cold_cand {
                if !kept.contains(&(i as i64)) {
                    cold_picks.push(i);
                    if cold_picks.len() >= self.cold_slots {
                        break;
                    }
                }
            }
            if !cold_picks.is_empty() {
                let keep_n = n.saturating_sub(cold_picks.len());
                items.truncate(keep_n);
                out_scores.truncate(keep_n);
                kinds.truncate(keep_n);
                for i in cold_picks {
                    items.push(i as i64);
                    out_scores.push(cs[i]);
                    kinds.push("cold_content".into());
                }
            }
        }
        (items, out_scores, kinds)
    }
}

impl EngineState {
    /// Full-catalog content-similarity scores for `owned` — port of
    /// `item_features.content_scores` (profile = Σ owned rows, score = F·profile;
    /// rows are L2-normalized at fit so this is a cosine-weighted sum).
    fn content_scores(&self, owned: &[i64]) -> Vec<f64> {
        let n_items = self.n_items;
        if self.content_nfeat == 0 || self.content_data.is_empty() || owned.is_empty() {
            return vec![0.0; n_items];
        }
        let mut profile = vec![0f64; self.content_nfeat];
        for &item in owned {
            let item = item as usize;
            if item + 1 < self.content_indptr.len() {
                for k in self.content_indptr[item] as usize..self.content_indptr[item + 1] as usize {
                    profile[self.content_indices[k] as usize] += self.content_data[k] as f64;
                }
            }
        }
        if profile.iter().all(|&x| x == 0.0) {
            return vec![0.0; n_items];
        }
        let nrow = self.content_indptr.len().saturating_sub(1);
        let mut scores = vec![0f64; n_items];
        for j in 0..nrow.min(n_items) {
            let s = self.content_indptr[j] as usize;
            let e = self.content_indptr[j + 1] as usize;
            let mut acc = 0f64;
            for k in s..e {
                acc += self.content_data[k] as f64 * profile[self.content_indices[k] as usize];
            }
            scores[j] = acc;
        }
        scores
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EngineState>()?;
    m.add_function(wrap_pyfunction!(build_engine, m)?)?;
    Ok(())
}
