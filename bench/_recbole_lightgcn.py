"""Train LightGCN on a pre-split (benchmark) dataset and dump top-50 per test
user. Runs in .venv-recbole. Driven by bench/run_lightgcn_warming.py, which
writes the {dataset}.{train,valid,test}.inter files (the warming subsample) and
scores the predictions with kindling's own aggregate() — so the LightGCN line is
comparable to the other warming rows.

Usage: .venv-recbole/bin/python bench/_recbole_lightgcn.py <data_path> <dataset> <out.json>
"""

import json
import sys
import time

import numpy as np
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_seed
from recbole.utils.case_study import full_sort_topk

data_path, dataset, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

cfg = {
    "data_path": data_path,
    "benchmark_filename": ["train", "valid", "test"],
    "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
    "metrics": ["NDCG"], "topk": [10], "valid_metric": "NDCG@10",
    "epochs": 100, "train_batch_size": 4096, "learning_rate": 0.001,
    "embedding_size": 64, "n_layers": 2, "reg_weight": 1e-4,
    "stopping_step": 10, "eval_batch_size": 4096,
    "seed": 2020, "reproducibility": True, "device": "cpu", "show_progress": False,
}
config = Config(model="LightGCN", dataset=dataset, config_dict=cfg)
init_seed(config["seed"], config["reproducibility"])
ds = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, ds)
model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

t0 = time.perf_counter()
trainer.fit(train_data, valid_data, saved=False, show_progress=False)
fit_s = time.perf_counter() - t0

uid_field, iid_field = ds.uid_field, ds.iid_field
test_uids = np.unique(test_data._dataset.inter_feat[uid_field].numpy())
topk: dict[str, list[str]] = {}
for s in range(0, len(test_uids), 2048):
    chunk = test_uids[s : s + 2048]
    _, iid = full_sort_topk(chunk, model, test_data, k=50, device=config["device"])
    for u, items in zip(chunk, iid.cpu().numpy()):
        topk[str(ds.id2token(uid_field, u))] = [str(x) for x in ds.id2token(iid_field, items)]

with open(out_path, "w") as fh:
    json.dump({"fit_seconds": round(fit_s, 1), "topk": topk}, fh)
print(f"LightGCN fit={fit_s:.1f}s test_users={len(test_uids)}", flush=True)
