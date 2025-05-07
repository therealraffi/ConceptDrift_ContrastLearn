#!/usr/bin/env python
# coding: utf-8

# In[18]:


import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from my_models import ContrastiveGraphDataset, load_graph, to_heterodata
from relation_graph import RelationGraph
import joblib


# In[20]:


meta = pd.read_csv("/scratch/rhm4nj/ml-sec/data/dataset/apk_metadata.csv", parse_dates=["datetime"]).set_index("apk_id")
rg = RelationGraph(
    rel_path="/scratch/rhm4nj/ml-sec/APIGraph/src/res/relations.txt",
    ent_path="/scratch/rhm4nj/ml-sec/APIGraph/src/res/entities.txt",
    type_map={'package':1, 'class':2, 'method':3, 'permission':4},
    rel_map={1:'function_of',2:'class_of',3:'inheritance',4:'uses_parameter',5:'returns',
             6:'throws',7:'alternative',8:'conditional',9:'refers_to',10:'uses_permission'}
)

sample_path = next(Path("/scratch/rhm4nj/ml-sec/data/subgraphs").glob("*.gpickle"))
sample_data = to_heterodata(pickle.load(open(sample_path, "rb")), rg)
metadata = sample_data.metadata()

dataset = ContrastiveGraphDataset(
    graph_dir="/scratch/rhm4nj/ml-sec/data/subgraphs",
    meta=meta,
    rg=rg,
    metadata=metadata,
    num_pairs=5000,
    sample_pairs=5000,
    pair_file="/scratch/rhm4nj/ml-sec/data/dataset/pairs.pkl",
    save_pairs=False
)

label_map = {'strong': 0, 'weak': 1, 'benign_malware': 2}
X, y = [], []


# In[ ]:


for g1, g2, label, _ in dataset:
    pool = lambda xdict: torch.cat([xdict[nt].mean(dim=0) for nt in xdict], dim=0)
    x1 = pool(g1.x_dict)
    x2 = pool(g2.x_dict)
    x = torch.cat([x1, x2]).numpy()
    X.append(x)
    y.append(label_map[label])

X = np.stack(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf_raw = SVC(kernel="linear", C=1.0)
clf_raw.fit(X_raw_train, y_raw_train)
joblib.dump(clf_raw, "svm_raw.joblib")

clf_gnn = SVC(kernel="linear", C=1.0)
clf_gnn.fit(X_gnn_train, y_gnn_train)
joblib.dump(clf_gnn, "svm_gnn.joblib")

print(classification_report(y_test, pred, target_names=list(label_map.keys())))

