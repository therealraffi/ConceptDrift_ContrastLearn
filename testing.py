#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import yaml
import torch
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report

from my_models import *
from relation_graph import RelationGraph
from joblib import load


# In[ ]:


with open("eval_config.yml") as f:
    cfg = yaml.safe_load(f)

meta = pd.read_csv(cfg["dataset"]["meta_csv"], parse_dates=["datetime"]).set_index("apk_id")
graph_dir = Path(cfg["dataset"]["graph_dir"])

rg = RelationGraph(
    rel_path=cfg["relation_graph"]["rel_path"],
    ent_path=cfg["relation_graph"]["ent_path"],
    type_map=cfg["relation_graph"]["type_map"],
    rel_map=cfg["relation_graph"]["rel_map"]
)

sample_path = next(graph_dir.glob("*.gpickle"))
sample_data = to_heterodata(pickle.load(open(sample_path, "rb")), rg)
metadata = sample_data.metadata()

model = HeteroGNN(metadata).to("cuda")
model.load_state_dict(torch.load(cfg["evaluation"]["model_path"]))
model.eval()

label_map = cfg["evaluation"]["label_map"]
inv_label_map = {v: k for k, v in label_map.items()}
delta = cfg["evaluation"].get("margin", 0.0)

# Compute centroids from a sample of benign + malware graphs
benign_ids = [apk for apk in meta.index if meta.loc[apk, "class"] == "benign"]
malware_ids = [apk for apk in meta.index if meta.loc[apk, "class"] != "benign"]

benign_ids = [i for i in benign_ids if (graph_dir / f"{i}.gpickle").exists()]
malware_ids = [i for i in malware_ids if (graph_dir / f"{i}.gpickle").exists()]

benign_sample = benign_ids[:200]
malware_sample = malware_ids[:200]

def pooled_emb(apk):
    g = load_graph(graph_dir / f"{apk}.gpickle", rg, metadata).to("cuda")
    with torch.no_grad():
        z = model(g.x_dict, g.edge_index_dict)
        return torch.cat([z[nt].mean(dim=0, keepdim=True) for nt in metadata[0]], dim=1).squeeze(0).cpu().numpy()

mu_benign = np.stack([pooled_emb(apk) for apk in benign_sample])
mu_malware = np.stack([pooled_emb(apk) for apk in malware_sample])

mu_b = mu_benign.mean(axis=0)
mu_m = mu_malware.mean(axis=0)


# In[ ]:


for year in cfg["evaluation"]["years"]:
    ids = [i for i, row in meta.iterrows() if row.datetime.year == year]
    ids = ids[:cfg["evaluation"]["samples_per_year"]]
    ids = [i for i in ids if (graph_dir / f"{i}.gpickle").exists()]

    X_test, y_true = [], []

    for apk in ids:
        z = pooled_emb(apk)
        label = meta.loc[apk, "class"]
        y_true.append(label_map["benign_malware"] if label == "benign" else label_map["strong"])
        X_test.append(z)

    X_test = np.stack(X_test)
    y_true = np.array(y_true)

    d_b = np.linalg.norm(X_test - mu_b, axis=1)
    d_m = np.linalg.norm(X_test - mu_m, axis=1)
    y_pred = np.where(d_m + delta < d_b, label_map["strong"], label_map["benign_malware"])

    print(f"\nYear {year}")
    print(classification_report(y_true, y_pred, target_names=list(label_map.keys()), zero_division=0))

