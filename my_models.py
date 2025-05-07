import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, global_mean_pool
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData, Batch
from torch.utils.tensorboard import SummaryWriter

import pickle
import random
from pathlib import Path
from collections import defaultdict
import numpy as np

from relation_graph import RelationGraph

class HeteroGNN(nn.Module):
    def __init__(self, metadata, hidden_channels=64, out_channels=128):
        super().__init__()
        self.metadata = metadata  # (node_types, edge_types)

        self.convs = nn.ModuleList([
            HeteroConv(
                {
                    edge_type: GCNConv(-1, hidden_channels)
                    for edge_type in metadata[1]
                },
                aggr='sum'
            ),
            HeteroConv(
                {
                    edge_type: GCNConv(hidden_channels, out_channels)
                    for edge_type in metadata[1]
                },
                aggr='sum'
            )
        ])

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        return x_dict

    def pool(self, x_dict, batch_dict):
        pooled = []
        for node_type in x_dict:
            pooled.append(global_mean_pool(x_dict[node_type], batch_dict[node_type]))
        return torch.cat(pooled, dim=1)  # [batch_size, total_feature_dim]

class ContrastiveGraphDataset(Dataset):
    def __init__(self, graph_dir, meta, rg, metadata, num_pairs=10000, sample_pairs=1000, pair_file=None, save_pairs=False):
        assert num_pairs >= sample_pairs
        self.meta = meta
        self.sample_pairs = sample_pairs
        self.graph_dir = Path(graph_dir)
        self.all_ids = [f.stem for f in self.graph_dir.glob("*.gpickle")]
        self.class_to_ids = defaultdict(list)
        self.rg = rg
        self.metadata = metadata

        for aid in self.all_ids:
            if aid not in meta.index:
                continue
            label = meta.loc[aid, "class"]
            self.class_to_ids[label].append(aid)

        self.max_possible_pairs = self._estimate_max_possible_pairs()
        print(f"max possible pairs (approx): {self.max_possible_pairs:,}")

        if pair_file and Path(pair_file).exists():
            print(f"loading pairs from {pair_file}")
            with open(pair_file, "rb") as f:
                self.pairs = pickle.load(f)
            self.pairs = random.sample(self.pairs, min(num_pairs, len(self.pairs)))
            print("Num pairs", len(self.pairs))
        else:
            print("building new pairs:")
            self.pairs = self.build_pairs(num_pairs)
            if save_pairs and pair_file:
                with open(pair_file, "wb") as f:
                    pickle.dump(self.pairs, f)

        self.sample_epoch_pairs()  # initialize sampled_pairs for first epoch

    def sample_epoch_pairs(self):
        self.sampled_pairs = random.sample(self.pairs, min(self.sample_pairs, len(self.pairs)))

    def _estimate_max_possible_pairs(self):
        n = len(self.all_ids)
        return n * (n - 1) // 2

    def build_pairs(self, num_pairs=10000):
        pairs = []
        ids = [aid for aid in self.all_ids if aid in self.meta.index]
        benign_ids = self.class_to_ids.get('benign', [])
        malware_ids = [aid for cls, aids in self.class_to_ids.items() if cls != 'benign' for aid in aids]

        for _ in tqdm(range(num_pairs)):
            aid1 = random.choice(ids)
            cls1 = self.meta.loc[aid1, "class"]

            if cls1 == "benign":
                aid2 = random.choice(malware_ids)
                label = "benign_malware"
            else:
                same_family = self.class_to_ids[cls1]
                if len(same_family) > 1 and random.random() < 0.5:
                    candidates = [i for i in same_family if i != aid1]
                    if candidates:
                        aid2 = random.choice(candidates)
                        label = "strong"
                    else:
                        aid2 = random.choice([i for i in ids if self.meta.loc[i, "class"] != cls1])
                        label = "weak"
                else:
                    aid2 = random.choice([i for i in ids if self.meta.loc[i, "class"] != cls1])
                    label = "weak"

            pairs.append((aid1, aid2, label))
        return pairs

    def __len__(self):
        return len(self.sampled_pairs)

    def __getitem__(self, idx):
        aid1, aid2, label = self.sampled_pairs[idx]
        g1 = load_graph(self.graph_dir / f"{aid1}.gpickle", self.rg, self.metadata)
        g2 = load_graph(self.graph_dir / f"{aid2}.gpickle", self.rg, self.metadata)

        t1 = self.meta.loc[aid1, "datetime"]
        t2 = self.meta.loc[aid2, "datetime"]
        delta_t = abs((t1 - t2).total_seconds()) / (60 * 60 * 24)
        return g1, g2, label, delta_t

def hcl_loss(z1, z2, label, delta_t, λ1=1.0, λ2=1.0, λ3=1.0, τ=1.0, m0=0.4, m2=1.0, β=0.1):
    """Hierarchical Contrastive Loss with time-aware margin."""
    label = np.array(label)  # <-- convert list of strings to array
    sim = F.cosine_similarity(z1, z2)
    losses = []

    mask_strong = (label == 'strong')
    if mask_strong.any():
        margin = 1 - torch.exp(-delta_t[mask_strong] / τ)
        losses.append(λ1 * F.relu(margin - sim[mask_strong]).mean())

    mask_weak = (label == 'weak')
    if mask_weak.any():
        margin = m0 * torch.exp(-β * delta_t[mask_weak])
        losses.append(λ2 * F.relu(margin - sim[mask_weak]).mean())

    mask_diff = (label == 'benign_malware')
    if mask_diff.any():
        losses.append(λ3 * F.relu(sim[mask_diff] - m2).mean())

    return sum(losses)


def load_graph(path, rg, metadata):
    with open(path, 'rb') as f:
        nxg = pickle.load(f)

    data = HeteroData()
    nodes_by_type = rg.nodes_by_type(nxg)

    for ntype, nodes in nodes_by_type.items():
        data[ntype].x = torch.randn(len(nodes), 32)
        node_map = {nid: i for i, nid in enumerate(nodes)}

    edge_types_present = set()
    for (src_type, rel_type, dst_type), (src, dst) in rg.edges_by_type(nxg).items():
        src_map = {nid: i for i, nid in enumerate(nodes_by_type[src_type])}
        dst_map = {nid: i for i, nid in enumerate(nodes_by_type[dst_type])}
        edge_index = torch.tensor([
            [src_map[s] for s in src],
            [dst_map[d] for d in dst]
        ], dtype=torch.long)
        data[(src_type, rel_type, dst_type)].edge_index = edge_index
        edge_types_present.add((src_type, rel_type, dst_type))

    for edge_type in metadata[1]:  # (src_type, rel_type, dst_type)
        if edge_type not in edge_types_present:
            data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)

    return data

def collate_fn(batch):
    g1s, g2s, labels, delta_ts = zip(*batch)
    batch1 = Batch.from_data_list(g1s)
    batch2 = Batch.from_data_list(g2s)
    label_tensor = torch.tensor([{'strong': 0, 'weak': 1, 'benign_malware': 2}[l] for l in labels])
    delta_ts = torch.tensor(delta_ts, dtype=torch.float32)
    return batch1, batch2, label_tensor, delta_ts

def to_heterodata(nxg, rg, feature_dim=32):
    data = HeteroData()
    nodes_by_type = rg.nodes_by_type(nxg)
    for ntype, nodes in nodes_by_type.items():
        data[ntype].x = torch.randn(len(nodes), feature_dim)
    for (src_type, rel_type, dst_type), (src, dst) in rg.edges_by_type(nxg).items():
        src_map = {nid: i for i, nid in enumerate(nodes_by_type[src_type])}
        dst_map = {nid: i for i, nid in enumerate(nodes_by_type[dst_type])}
        edge_index = torch.tensor([
            [src_map[s] for s in src],
            [dst_map[d] for d in dst]
        ], dtype=torch.long)
        data[(src_type, rel_type, dst_type)].edge_index = edge_index
    return data