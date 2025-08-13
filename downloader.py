import pandas as pd
import json
import sys
import subprocess
import os
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

setup_files = False
API_KEY = os.getenv("DOWNLOAD_KEY")
ENDPOINT = 'https://androzoo.uni.lu/api/download'
cdir = "/scratch/rhm4nj/ml-sec"
logdir = Path(cdir) / "logs"
outdir = Path(cdir) / "apks"

if setup_files:
    input_csv = 'data/latest.csv'
    output_json = 'data/mdtosha.json'
    
    tbl = pd.read_csv(input_csv, usecols=['md5', 'sha1', 'sha256'])
    tbl = tbl.drop_duplicates('md5', keep='first')
    mp = tbl.set_index('md5')[['sha1', 'sha256']].to_dict('index')
    with open(output_json, 'w') as f:
        json.dump(mp, f, indent=2)

    print(f"Unique md5s:     {tbl['md5'].nunique()}")
    print(f"Unique sha256s:  {tbl['sha256'].nunique()}")
    print(f"done")

json_mdtosha = Path(cdir) / 'data/mdtosha.json'

print("Loading data!")
with open(json_mdtosha) as f:
    mp = json.load(f)
print("Done loading!", len(mp))

logdir.mkdir(parents=True, exist_ok=True)
outdir.mkdir(parents=True, exist_ok=True)

shas = [v['sha256'] for v in mp.values() if 'sha256' in v]

for sha in tqdm(shas, desc="Downloading APKs"):
    output_filepath = outdir / f"{sha}.apk"
    
    if output_filepath.exists():
        continue
    
    curl_command = [
        "curl", "-G",
        "-o", str(output_filepath),
        "-d", f"apikey={API_KEY}",
        "-d", f"sha256={sha}",
        ENDPOINT
    ]
    
    try:
        subprocess.run(curl_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {sha}: {e}")
    except FileNotFoundError:
        print("Error: 'curl' command not found.")
        sys.exit(1)

print("All APK downloads attempted.")