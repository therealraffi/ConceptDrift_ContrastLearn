#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

apk_dir = Path("/scratch/rhm4nj/ml-sec/data/apks")
out_csv = Path("/scratch/rhm4nj/ml-sec/data/dataset/apk_metadata.csv")

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

