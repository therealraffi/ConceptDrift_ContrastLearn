import csv
import os
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import networkx as nx
from androguard.misc import AnalyzeAPK
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from relation_graph import RelationGraph

apk_dir = Path("../data/apks")
out_csv = Path("../data/dataset/apk_metadata.csv")
dataset_folder = "../APIGraph/src/res"

malware_classes = [
    'skymobi', 'youmi', 'smsreg', 'lotoor', 'artemis', 'secapk', 'dowgin',
    'airpush', 'fusob', 'triada', 'fakeinst', 'wapsx', 'boxer', 'kuguo',
    'adwo', 'appquanta', 'plankton', 'smspay', 'smsspy', 'mecor', 'smforw',
    'utchi', 'opfake', 'umpay', 'dnotua', 'waps', 'admogo', 'hiddad',
    'hiddenapp', 'hiddenads'
]

# gather apk names (stem of filename)
apk_paths = list(apk_dir.glob("*.apk"))
apk_ids = [p.stem for p in apk_paths]

random.seed(42)
random.shuffle(apk_ids)

start = datetime(2013, 1, 1)
end = datetime(2018, 12, 31)

def random_time():
    delta = end - start
    rand_sec = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=rand_sec)

# label distribution
n_total = len(apk_ids)
n_benign = int(n_total * 0.75)
n_malware = n_total - n_benign

labels = (
    ['benign'] * n_benign +
    [random.choice(malware_classes) for _ in range(n_malware)]
)
random.shuffle(labels)

with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["apk_id", "datetime", "class"])
    for apk_id, cls in zip(apk_ids, labels):
        dt = random_time().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([apk_id, dt, cls])

print(f"Wrote metadata for {n_total} APKs to {out_csv}")

### Process APK
all_apks = sorted([str(p) for p in Path(apk_dir).glob("*.apk")])
for apk_fp in tqdm(all_apks, desc="Processing APKs"):
    apk_id = os.path.basename(apk_fp).replace(".apk", "")
    apk_obj, dvm_obj, analysis_obj = AnalyzeAPK(apk_fp)

    type_map = {'package':1,'class':2,'method':3,'permission':4}
    relation_map = {
        1:'function_of',2:'class_of',3:'inheritance',
        4:'uses_parameter',5:'returns',6:'throws',
        7:'alternative',8:'conditional',9:'refers_to',
        10:'uses_permission'
    }

    rg = RelationGraph(
        f'{dataset_folder}/relations.txt',
        f'{dataset_folder}/entities.txt',
        type_map,
        relation_map
    )

    subg = rg.apk_subgraph(analysis_obj)
    rg.save_subgraph(subg, f'../data/subgraphs/{apk_id}.gpickle')

print("All APKs processed.")