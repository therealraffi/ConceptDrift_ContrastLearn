# Contrastive Malware Classification with Graph Embeddings

This project builds a malware classification system using contrastive learning over heterogeneous graphs. We compare a HeteroGNN trained with a time-aware hierarchical contrastive loss to a linear SVM baseline. The goal is to detect concept drift over time as malware evolves.

Note - I developed the code in notebooks first (see notebooks folder) then converted to python scripts.

## Setup
```bash
python -m venv env
source activate env/bin/activate
pip install -r requirements.txt

git clone https://github.com/seclab-fudan/APIGraph
```

To download apks, follow instructions in https://github.com/seclab-fudan/APIGraph

## Pre-Processing
```bash
cd APIGraph/src
python getAllEntities.py
python getAllRelations.py

cd ../../
python preprocess.py # need SLURM access - run on Rivanna
python process_apk.py # need SLURM access - run on Rivanna
python run_graphs.py # need SLURM access - run on Rivanna
```

## Training
python trainer.py

```bash
python trainer.py
python svm.py
```

To run Chen et. Al, see setup instructions in https://github.com/wagner-group/active-learning
