import networkx as nx
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from androguard.misc import AnalyzeAPK
import pickle
from relation_graph import RelationGraph
import sys
import os
import shutil

if len(sys.argv) != 2:
    print("Usage: python process_apk.py <path_to_apk>")
    sys.exit(1)
apk_fp = sys.argv[1]

apk_id = os.path.basename(apk_fp).replace(".apk", "")
apk_obj, dvm_obj, analysis_obj = AnalyzeAPK(apk_fp)

dataset_folder = "/scratch/rhm4nj/ml-sec/APIGraph/src/res"
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
