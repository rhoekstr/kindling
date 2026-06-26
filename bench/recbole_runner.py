"""Run RecBole baselines on the shared ml-1m split and export everything needed
for an apples-to-apples comparison with kindling.

Runs in the dedicated `.venv-recbole` (RecBole + torch). For each model it fits
(timed), evaluates with RecBole's own full-ranking metrics, and exports the
per-user top-10 predictions — so a single external scorer can grade every model
(RecBole baselines *and* kindling) the same way. Also exports the exact
train/test split (external ids) so kindling fits on identical data.

    .venv-recbole/bin/python bench/recbole_runner.py Pop ItemKNN EASE
    .venv-recbole/bin/python bench/recbole_runner.py BPR LightGCN

Outputs (under recbole_data/):
    split_train.csv         entity_id,item_id,rating,timestamp  (RecBole's train)
    split_truth.json        {user_ext: [test_item_ext, ...]}
    recbole_<model>.json    {metrics, fit_seconds, eval_seconds, predict_seconds,
                             topk: {user_ext: [item_ext x10]}}
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_seed
from recbole.utils.case_study import full_sort_topk

OUT = Path("recbole_data")
K = 10
SPLIT = {"split": {"RS": [0.8, 0.1, 0.1]}, "order": "RO", "group_by": "user", "mode": "full"}

# Per-model overrides. Non-iterative models train in one pass; the two trainable
# baselines use modest, standard configs (capped epochs + early stopping).
OVERRIDES = {
    "Pop": {},
    "ItemKNN": {"k": 100, "shrink": 0.0},
    "EASE": {"reg_weight": 250.0},
    "BPR": {"epochs": 100, "train_batch_size": 4096, "learning_rate": 0.001, "embedding_size": 64},
    "LightGCN": {
        "epochs": 100, "train_batch_size": 4096, "learning_rate": 0.001,
        "embedding_size": 64, "n_layers": 2, "reg_weight": 1e-4,
    },
}


def run(model_name: str) -> None:
    cfg_dict = {
        "data_path": str(OUT),
        "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
        "eval_args": SPLIT,
        "metrics": ["Recall", "NDCG", "MRR", "Hit", "Precision"],
        "topk": [K],
        "valid_metric": "NDCG@10",
        "eval_batch_size": 4096,
        "stopping_step": 10,
        "seed": 2020,
        "reproducibility": True,
        "device": "cpu",
        "show_progress": False,
        **OVERRIDES.get(model_name, {}),
    }
    config = Config(model=model_name, dataset="ml-1m", config_dict=cfg_dict)
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    t0 = time.perf_counter()
    trainer.fit(train_data, valid_data, saved=False, show_progress=False)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    metrics = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
    eval_s = time.perf_counter() - t0

    uid_field, iid_field = dataset.uid_field, dataset.iid_field

    # Ground-truth test items per user (external ids) — exported once.
    truth_path = OUT / "split_truth.json"
    if not truth_path.exists():
        tf = test_data._dataset.inter_feat
        uids = tf[uid_field].numpy()
        iids = tf[iid_field].numpy()
        truth: dict[str, list[str]] = {}
        u_ext = dataset.id2token(uid_field, uids)
        i_ext = dataset.id2token(iid_field, iids)
        for u, i in zip(u_ext, i_ext):
            truth.setdefault(str(u), []).append(str(i))
        truth_path.write_text(json.dumps(truth))
        # Train split for kindling (external ids).
        trf = train_data._dataset.inter_feat
        import csv
        with open(OUT / "split_train.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["entity_id", "item_id", "rating", "timestamp"])
            tu = dataset.id2token(uid_field, trf[uid_field].numpy())
            ti = dataset.id2token(iid_field, trf[iid_field].numpy())
            tr = trf["rating"].numpy()
            for u, i, r in zip(tu, ti, tr):
                w.writerow([u, i, float(r), 0])

    # Top-10 predictions per test user.
    test_uids = np.unique(test_data._dataset.inter_feat[uid_field].numpy())
    topk: dict[str, list[str]] = {}
    t0 = time.perf_counter()
    BATCH = 2048
    for s in range(0, len(test_uids), BATCH):
        chunk = test_uids[s : s + BATCH]
        _, topk_iid = full_sort_topk(chunk, model, test_data, k=K, device=config["device"])
        for u_int, items_int in zip(chunk, topk_iid.cpu().numpy()):
            u_ext = str(dataset.id2token(uid_field, u_int))
            topk[u_ext] = [str(x) for x in dataset.id2token(iid_field, items_int)]
    predict_s = time.perf_counter() - t0

    out = {
        "model": model_name,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "fit_seconds": round(fit_s, 3),
        "eval_seconds": round(eval_s, 3),
        "predict_seconds": round(predict_s, 3),
        "n_test_users": int(len(test_uids)),
        "topk": topk,
    }
    (OUT / f"recbole_{model_name}.json").write_text(json.dumps(out))
    print(
        f"{model_name:10s} NDCG@10={metrics.get('ndcg@10'):.4f} Recall@10={metrics.get('recall@10'):.4f} "
        f"fit={fit_s:.1f}s predict={predict_s:.1f}s users={len(test_uids)}",
        flush=True,
    )


if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads()))
    models = sys.argv[1:] or ["Pop", "ItemKNN", "EASE", "BPR", "LightGCN"]
    for m in models:
        run(m)
