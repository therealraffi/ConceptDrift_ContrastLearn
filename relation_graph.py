import networkx as nx
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import random

from androguard.misc import AnalyzeAPK
import pickle

class RelationGraph:
    def __init__(self, rel_path, ent_path, type_map, rel_map):
        self.G = nx.DiGraph()
        self.code2type = {v:k for k,v in type_map.items()}
        self.rel_map    = rel_map
        self.ent2id     = {}
        self.id2ent     = {}
        self.node_type  = {}

        with open(ent_path) as f:
            for idx,(ent,tcode) in enumerate(csv.reader(f), start=1):
                self.ent2id[ent]   = idx
                self.id2ent[idx]   = ent
                self.node_type[idx] = self.code2type.get(int(tcode),'unknown')

        with open(rel_path) as f:
            for s,r,o in csv.reader(f):
                u,v,rid = int(s), int(o), int(r)
                self.G.add_edge(u, v, rel=rid)

    def apk_subgraph(self, analysis):
        used = set()
        for m in analysis.get_methods():
            for cls_node, meth_node, _ in m.get_xref_to():
                cls_name = cls_node.name
                if not cls_name.startswith("Landroid"): 
                    continue
                mname = meth_node.method.get_name()
                fq  = f"{cls_name.strip('L;').replace('/','.')}.{mname}"
                used.add(fq)

        ids = [ self.ent2id[n] for n in used if n in self.ent2id ]
        return self.G.subgraph(ids).copy()

    def nodes_by_type(self, sub=None):
        G = sub or self.G
        d = defaultdict(list)
        for n in G.nodes():
            d[self.node_type[n]].append(n)
        return dict(d)

    def edges_by_type(self, sub=None):
        G = sub or self.G
        out = defaultdict(lambda: ([], []))
        for u,v,d in G.edges(data=True):
            key = ( self.node_type[u],
                    self.rel_map[d['rel']],
                    self.node_type[v] )
            src, dst = out[key]
            src.append(u); dst.append(v)
        return {k: tuple(v) for k,v in out.items()}

    def to_hetero_inputs(self, sub=None):
        return self.nodes_by_type(sub), self.edges_by_type(sub)

    def visualize(self, sub=None, n=50, node_size=800):
        G = sub or self.G
        nodes = list(G.nodes())
        if len(nodes) > n:
            nodes = random.sample(nodes, n)
        H = G.subgraph(nodes)
        types = sorted({self.node_type[n] for n in H})
        cmap_n = plt.cm.tab10.colors
        ncol = {t: cmap_n[i % len(cmap_n)] for i, t in enumerate(types)}
        nc = [ncol[self.node_type[n]] for n in H]
        rels = sorted({d['rel'] for _, _, d in H.edges(data=True)})
        cmap_e = plt.cm.Set2.colors
        ecol = {r: cmap_e[i % len(cmap_e)] for i, r in enumerate(rels)}
        ec = [ecol[d['rel']] for _, _, d in H.edges(data=True)]
        pos = nx.spring_layout(H)
        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(H, pos, node_color=nc, node_size=node_size)
        nx.draw_networkx_labels(H, pos, font_color='white', font_size=8)
        nx.draw_networkx_edges(H, pos, edge_color=ec, arrowsize=12)
        nleg = [Patch(color=c, label=t) for t, c in ncol.items()]
        eleg = [Line2D([0], [0], color=ecol[r], lw=2, label=self.rel_map[r]) for r in rels]
        plt.legend(handles=nleg + eleg, loc='best')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def save_subgraph(self, subgraph, path):
        with open(path, 'wb') as f:
            pickle.dump(subgraph, f)

    def load_subgraph(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)