#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json
import sys
import subprocess

import networkx as nx
import csv
import matplotlib.pyplot as plt

from tqdm import tqdm
import shutil
import os
from pathlib import Path
from itertools import islice

import json
import os
import time


# In[2]:


setup_files = False
if setup_files:
    input_csv = 'data/latest.csv'
    output_json = 'data/mdtosha.json'

    tbl = pd.read_csv(input_csv, usecols=['md5', 'sha1', 'sha256'])
    tbl = tbl.drop_duplicates('md5', keep='first')
    mp = tbl.set_index('md5')[['sha1','sha256']].to_dict('index')
    with open(output_json, 'w') as f:
        json.dump(mp, f, indent=2)

    print(f"Unique md5s:     {tbl['md5'].nunique()}")
    print(f"Unique sha256s:  {tbl['sha256'].nunique()}")
    print(f"done")


# In[4]:


json_mdtosha = 'data/mdtosha.json'

print("Loading data!")
mp = json.load(open(json_mdtosha))
print("Done loading!", len(mp))


# In[ ]:


import os
import subprocess
import time
from pathlib import Path
from itertools import islice
from tqdm import tqdm

API_KEY = '...'  # fill in
ENDPOINT = 'https://androzoo.uni.lu/api/download'
cdir = "/scratch/rhm4nj/ml-sec"
logdir = f"{cdir}/logs"
outdir = f"{cdir}/apks"
BATCH_SIZE = 25
MAX_JOBS = 100

Path(logdir).mkdir(parents=True, exist_ok=True)
Path(outdir).mkdir(parents=True, exist_ok=True)
if os.path.exists(os.path.join(cdir, "stop")):
    os.remove(os.path.join(cdir, "stop"))

def active_job_count():
    cmd = ["squeue", "-h", "-u", os.getenv("USER")]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return len(result.stdout.strip().splitlines())

def batch(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

def write_job_script(batch_id, sha_list):
    script_path = f"{logdir}/batch_{batch_id}.sh"
    with open(script_path, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --account=cral
#SBATCH --partition=standard
#SBATCH --job-name=dl_batch_{batch_id}
#SBATCH --output={logdir}/batch_{batch_id}.out
#SBATCH --error={logdir}/batch_{batch_id}.err
#SBATCH --time=00:15:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1

""")
        for sha in sha_list:
            f.write(f"""curl -G -o {outdir}/{sha}.apk \\
    -d apikey={API_KEY} \\
    -d sha256={sha} \\
    {ENDPOINT}

""")
    return script_path

def download_hashes_batched(limit, md5rows, mp):
    seen = set()
    needed = []
    for h in md5rows.md5:
        if len(needed) >= limit:
            break
        rec = mp.get(h)
        if not rec:
            continue
        sha = rec['sha256']
        if sha in seen or os.path.exists(f"{outdir}/{sha}.apk"):
            continue
        seen.add(sha)
        needed.append(sha)

    print(f"To download: {len(needed)}")
    batches = list(batch(needed, BATCH_SIZE))

    for i, batch_list in tqdm(enumerate(batches), total=len(batches), desc="Submitting batches"):
        while active_job_count() >= MAX_JOBS:
            time.sleep(5)
        script = write_job_script(i, batch_list)
        subprocess.run(["sbatch", script])
        if "stop" in os.listdir(cdir):
            print("Detected stop signal — halting submissions.")
            break

    print("All jobs submitted.")

