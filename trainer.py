#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, global_mean_pool

import os
import pickle
import random
from pathlib import Path
from datetime import datetime
import random

from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData, Batch

from relation_graph import RelationGraph

import pandas as pd
from tqdm import tqdm

from collections import defaultdict
from tqdm import tqdm

import numpy as np

from torch.utils.tensorboard import SummaryWriter

from my_models import *
import yaml


# In[ ]:


with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
train_cfg = config["training"]

meta = pd.read_csv(paths["meta_csv"], parse_dates=["datetime"]).set_index("apk_id")
print("done loading metadata")

type_map = {'package':1, 'class':2, 'method':3, 'permission':4}
rel_map = {
    1:'function_of',2:'class_of',3:'inheritance',
    4:'uses_parameter',5:'returns',6:'throws',
    7:'alternative',8:'conditional',9:'refers_to',10:'uses_permission'
}
rg = RelationGraph(paths["rel_path"], paths["ent_path"], type_map, rel_map)
print("done loading relation graph")

sample_path = next(Path(paths["graph_dir"]).glob("*.gpickle"))
with open(sample_path, "rb") as f:
    nxg = pickle.load(f)
metadata = to_heterodata(nxg, rg).metadata()

dataset = ContrastiveGraphDataset(
    graph_dir=paths["graph_dir"],
    meta=meta,
    rg=rg,
    metadata=metadata,
    num_pairs=train_cfg["num_pairs"],
    sample_pairs=train_cfg["sample_pairs"],
    pair_file=paths["pair_file"],
    save_pairs=train_cfg["save_pairs"],
)

loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True, collate_fn=collate_fn)
model = HeteroGNN(metadata=metadata).to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])


# In[7]:


writer = SummaryWriter(log_dir=paths["log_dir"])
label_map = {'strong': 0, 'weak': 1, 'benign_malware': 2}
print_epochs = 10
max_epochs = 50
min_loss = 1000

for ep in tqdm(range(max_epochs), desc="Training"):
    model.train()
    dataset.sample_epoch_pairs()
    total = 0

    for b1, b2, lbls, dts in loader:
        b1, b2 = b1.to('cuda'), b2.to('cuda')
        li = torch.tensor([label_map[l] for l in lbls], device='cuda')
        dts = dts.to('cuda')

        z1 = model.pool(model(b1.x_dict, b1.edge_index_dict), b1.batch_dict)
        z2 = model.pool(model(b2.x_dict, b2.edge_index_dict), b2.batch_dict)

        loss = hcl_loss(z1, z2, lbls, dts,
                        λ1=config["loss"]["lambda1"],
                        λ2=config["loss"]["lambda2"],
                        λ3=config["loss"]["lambda3"],
                        τ=config["loss"]["tau"],
                        m0=config["loss"]["m0"],
                        m2=config["loss"]["m2"],
                        β=config["loss"]["beta"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()

    writer.add_scalar("Loss/train", total, ep)
    if ep % print_epochs == 0 or ep == max_epochs - 1:
        print(f"Epoch {ep} | Loss: {total:.5f}")

    if total < min_loss:
        min_loss = loss
        torch.save(model.state_dict(), paths['ckpt_path'])
        print("Saving - epoch", ep, "loss", total)


writer.close()

