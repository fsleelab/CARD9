import os
import matplotlib.pyplot as plt
import umap
import hdbscan
from hdbscan.prediction import approximate_predict
import numpy as np
import pandas as pd
import pickle
from fine_behaviors_analysis.final_train_models import preprocess_data, load_unlabeled_mice
from fine_behaviors_analysis.txt_reader import read_txt
import json
from argparse import ArgumentParser

def _load_pickled_model(modelPath):
    with open(modelPath, 'rb') as f:
        reducer = pickle.load(f)
    return reducer

def predict(rootDir, projects, umapFile, hdbFile, paramFile, labelMap):
    reducer = _load_pickled_model(umapFile)
    hdbModel = _load_pickled_model(hdbFile)
    param_dict = read_txt(paramFile)
    param_dict['df_choice'] = 'distance'
    param_dict['n_components'] = 2
    with open(labelMap, 'r') as f:
        labelMap = json.load(f)

    mice = []
    for project in projects:
        dataDir = os.path.join(rootDir, project)
        if not os.path.isdir(dataDir):
            continue
        loaded_mice = load_unlabeled_mice(dataDir, 500)
        mice += loaded_mice
    for mouse in mice:
        single_mouse_report = {}
        data_roll, _ = preprocess_data([mouse], param_dict, False)
        embedding = reducer.transform(data_roll)
        pred_clusters, strengths = approximate_predict(hdbModel, embedding)
        values, counts = np.unique(pred_clusters, return_counts=True)
        for behavior in labelMap['behavior_labels'].keys():
            totalCounts = np.isin(pred_clusters, labelMap['behavior_labels'][behavior]).sum()
            single_mouse_report[behavior] = (totalCounts / pred_clusters.shape[0]) * 100
        df = pd.DataFrame(single_mouse_report, index =[0])
        save_path = os.path.split(mouse.file)[0]
        df.to_csv(f'{save_path}/revised_cluster_labels.csv')

if __name__ == '__main__':
    parser = ArgumentParser(description="Inference/behavior prediction parameters")
    parser.add_argument('--root_dir', help='Folder containing grouped Anipose project folders', default = './data')
    parser.add_argument('--projects', help='Name(s) of project folders to process; multiple folders should be entered as either space-seperated quoted string or a JSON', default='example_mouse')
    parser.add_argument('--umap_path', help='Path to fitted UMAP reducer .sav file', default = 'fine_behaviors_analysis/models/more_test_params_model.sav')
    parser.add_argument('--hdb_path', help='Path to fitted HDBSCAN model .sav file', default = 'fine_behaviors_analysis/models/more_test_params_hdbscan_model.sav')
    parser.add_argument('--param_file', help='Path to .txt file with parameters used in final_train_models.py', default = 'fine_behaviors_analysis/models/more_test_params.txt')
    parser.add_argument('--label_file', help='Path to .json file mapping cluster ID to behavior label, from get_behavior_hdbscanLabels.py', default = 'fine_behaviors_analysis/models/behavior_labels.json')
    args = parser.parse_args()

    if args.projects.endswith('.json'):
        with open(args.projects, 'r') as f:
            projects = json.load(f)
    else:
        projects = args.projects.split(' ')

    predict(args.root_dir, projects,            
            args.umap_path,
            args.hdb_path,
            args.param_file,
            args.label_file)
