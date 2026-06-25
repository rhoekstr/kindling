# Harness — evaluate and serve

Two production tools ship inside the wheel: an **evaluation harness** to prove
the model on your own data, and a **serving harness** to put it behind HTTP.
Both are reachable from the `kindling` console command and from Python.

---

## Evaluation harness

The same realistic-tier protocol the project validates itself with —
chronological split, full-catalog ranking, sliced by user history length —
packaged so you can point it at *your* interaction log. It answers the question
that decides adoption: **does kindling beat popularity (and the trained
baselines) on my data, and in which warmth regime?**

### CLI

```bash
# Your own interaction log (CSV with entity_id,item_id[,timestamp,rating]).
# Column aliases are recognized: user_id/customer_id → entity_id,
# product_id/article_id → item_id, ts/date → timestamp, …
kindling bench --data interactions.csv

# Add the trained baselines (needs the optional 'baselines' extra → implicit):
kindling bench --data interactions.csv --all-baselines --json report.json

# A built-in reference dataset, no download for the synthetic ones:
kindling bench --dataset synthetic-grocery --baselines popularity,als
```

Output is a per-warmth-bucket NDCG@k table; `--json` also writes the full
report (every metric, every bucket, every model).

```
mydata — NDCG@10 by user history (base=ease, n_items=4,812, fit=2.1s)
  active channels: trend, last_item, transitions, user_cf

bucket        n    kindling  popularity         als
-----------------------------------------------------
1-4         320     0.0414*     0.0231      0.0150
5-19        540     0.0876*     0.0302      0.0631
20+         310     0.1320      0.0488      0.1402*
all        1170     0.0840*     0.0331      0.0712

  * = best in row
```

### Python

```python
from kindling.harness import evaluate, format_report
from kindling.harness.data import resolve_dataset

split = resolve_dataset("interactions.csv", test_fraction=0.1)
report = evaluate(
    split.train, split.test, split.items,
    dataset=split.name, k=10,
    baselines=["popularity", "als", "bpr"],   # trained ones need `implicit`
)
print(format_report(report))
print(report.metric("kindling", bucket="1-4", name="ndcg@10"))
report.to_dict()   # JSON-serializable: every metric, bucket, and model
```

`evaluate()` fits `Engine(**engine_kwargs)` itself; pass engine overrides via
`engine_kwargs=...` (most knobs auto-gate — you rarely need to). Buckets default
to `1-4 / 5-19 / 20+` plus an implicit `all`; override with `buckets=...`.

Baselines available right now (trained ones require `pip install
kindling[baselines]`):

```python
from kindling.harness.baselines import available_baselines
available_baselines()   # ['popularity', 'item-knn', 'als', 'bpr'] if implicit is present
```

---

## Serving harness

Turn a fitted (or saved) engine into a small HTTP service. FastAPI + uvicorn
are an optional extra, imported lazily — the core wheel never needs them.

```bash
pip install 'kindling[serve]'
```

### Fit, save, serve from the CLI

```bash
kindling fit  --data interactions.csv --out engine.kindling
kindling serve --model engine.kindling --host 0.0.0.0 --port 8000
```

### Endpoints

| Method & path             | Body                          | For |
|---------------------------|-------------------------------|-----|
| `GET  /health`            | —                             | liveness, catalog size, base scorer |
| `GET  /`                  | —                             | service info + the activation plan |
| `POST /recommend`         | `{"entity_id": 42, "n": 10}`  | a **known** user |
| `POST /recommend_for_items` | `{"item_ids": [101, 205], "n": 10}` | a **new / anonymous** user (empty → popularity fallback) |
| `POST /recommend/batch`   | `{"requests": [ … ]}`         | a mix of the two shapes in one call |

`entity_id` accepts an int or a string; a catalog trained on integer ids still
resolves a `"42"` request. An unknown known-user id returns `404` (use
`/recommend_for_items` for users absent from training).

```bash
curl -s localhost:8000/recommend -d '{"entity_id": 42, "n": 5}'
# {"entity_id": 42, "recommendations": [{"item_id": 17, "score": 4.81, "base_kind": "ease"}, …]}
```

### Embedding the app

```python
from kindling import Engine
from kindling.serving import create_app

engine = Engine().fit(interactions)
app = create_app(engine)          # a FastAPI instance — mount it, add auth, etc.
# uvicorn app:app  /  or kindling.serving.serve(engine, port=8000)
```

`create_app` also accepts a path to a saved engine: `create_app("engine.kindling")`.

---

See [user-guide.md](user-guide.md) for the engine API and
[REFERENCE.md](REFERENCE.md) for the architecture and the activation gates.
