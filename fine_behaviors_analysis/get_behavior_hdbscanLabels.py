import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from fine_behaviors_analysis.txt_reader import read_txt
import json
from argparse import ArgumentParser

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Let the base class handle other objects
        return super().default(obj)

def show_labels(df, hdbscan_labels, important_labels = None, add_text = False, label_counts = None):
    import matplotlib as mpl
    all_labels = np.unique(hdbscan_labels)
    # if important_labels is a list, we make sure those get colored a bit differently than any other label
    if important_labels is not None:
        cmap = mpl.colormaps['winter'](np.linspace(0,1,len(all_labels)))
        cmap[np.isin(all_labels, important_labels)] = mpl.colormaps['autumn'](np.linspace(0,1,len(important_labels)))
    else:
        #cmap = mpl.colormaps['hsv'](np.linspace(0,1,len(all_labels))) # Change colormap to contrast with label coloring
        cmap = mpl.colormaps['cool'](np.linspace(0, 1, len(all_labels)))
    X = df[['x', 'y']].to_numpy()
    y = df['etho'].replace([0, '0.0'], '0').to_numpy() # Essential to make sure all values are of one dtype
    fig, axs = plt.subplots(1, 3, figsize=(18, 9), dpi=175) # (Hand scored, HDBSCAN labels, combined)
    top_layer = []
    for count, cluster_id in enumerate(all_labels):
        data_in_cluster = X[hdbscan_labels == cluster_id]
        centroid = data_in_cluster.mean(axis=0)
        if cluster_id == -1:
            color = (0.5, 0.5, 0.5, 0.5)  # Grey
            text = None
        else:
            color = cmap[count]
            text = str(cluster_id)
        if cluster_id in important_labels:
            if label_counts is not None:
                text = f'{text}:{label_counts[cluster_id]}'
            top_layer.append([centroid, text])
        for i in [1,2]:
            axs[i].scatter(data_in_cluster[:, 0],
                       data_in_cluster[:, 1],
                       s=0.1,
                       color=color)
    colorDict = {
        'rear': 'blue',
        'body groom': 'green',
        'face groom': 'lime',
        'hindpawGroom': 'orange',
        'hindpawScratch': 'red',
        'twitch': 'purple',
        '0': 'black',
    }
    for value in np.unique(y):
        y_values = X[y == value]
        if value == '0':
            axs[0].scatter(y_values[:, 0],
                           y_values[:, 1],
                           s=0.1,
                           color='black',
                           label=value,
                           edgecolors=None, alpha=0.08)
            continue
        try:
            color = colorDict[value]
        except:
            color = 'black'
        for i in [0,2]:
            axs[i].scatter(y_values[:,0],
                       y_values[:,1],
                       s=0.1,
                       color=color,
                       label=value,
                       edgecolors=None, alpha=0.2)
    if add_text:
        for centroid, text in top_layer:
            axs[2].text(centroid[0], centroid[1], text, color = 'black', fontsize=4)
    for i in [0,1,2]:
        axs[i].axis('off')
        axs[i].legend(markerscale=8)
    return fig

def get_behavior_labels(embedDataset,
                saveDir,
                paramFile,
                show_figure=True,
                save_figure=False,
                n_cpus=1):
    param_dict = read_txt(paramFile)
    saveName = os.path.split(paramFile)[-1].split('.')[0]

    df = pd.read_csv(embedDataset)

    X = df[['x', 'y']].to_numpy()
    y = df['etho'].replace([0, '0.0'], '0').to_numpy()  # Essential to make sure all values are of one dtype

    model = hdbscan.HDBSCAN(min_samples=param_dict['min_samples'],
                            min_cluster_size=param_dict['min_cluster_size'],
                            cluster_selection_method=param_dict['cluster_selection_method'],
                            cluster_selection_epsilon=param_dict['epsilon'],
                            core_dist_n_jobs=n_cpus,
                            prediction_data=True)
    hdbscan_labels = model.fit_predict(X)
    behavior_labels = {}
    behavior_label_counts = {}
    for behavior in np.unique(y):
        label_counts = {}
        included_labels = hdbscan_labels[y == behavior]
        included_labels = np.unique(included_labels)
        labels_to_remove = [-1]
        for label in included_labels:
            counts = sum(hdbscan_labels[y == behavior] == label)
            if counts <= 2:
                labels_to_remove.append(label)
            else:
                label_counts[label] = counts
        included_labels = included_labels[~np.isin(included_labels, labels_to_remove)]
        behavior_labels[behavior] = included_labels
        behavior_label_counts[behavior] = label_counts
    in_too_many_behaviors = []
    for label in hdbscan_labels:
        if all(label in behavior_labels[b] for b in behavior_labels.keys()):
            in_too_many_behaviors.append(label)
    for behavior, labels in behavior_labels.items():
        labels = labels[~np.isin(labels, in_too_many_behaviors)]
        behavior_labels[behavior] = labels
        if show_figure:
            fig = show_labels(df, hdbscan_labels, important_labels = labels, add_text=True, label_counts = behavior_label_counts[behavior])
            fig.suptitle(behavior)
            fig.show()
            plt.show()
        if save_figure:
            fig = show_labels(df, hdbscan_labels, important_labels = labels, add_text=True, label_counts = behavior_label_counts[behavior])
            fig.suptitle(behavior)
            fig.savefig(f'{saveDir}/{behavior}_labels.jpg')
    for key, value in behavior_labels.items():
        behavior_labels[key] = list(value)
    param_dict['behavior_labels'] = behavior_labels
    json_out = f'{saveDir}/behavior_labels.json'
    with open(json_out, 'w') as json_file:
        json.dump(param_dict, json_file, cls=NumpyEncoder, indent=4)
    return json_out

def show_final_clusters(embedDataset,
                saveDir,
                paramFile,
                jsonFile,
                n_cpus=1):
    param_dict = read_txt(paramFile)
    df = pd.read_csv(embedDataset)
    X = df[['x', 'y']].to_numpy()
    y = df['etho'].replace([0, '0.0'], '0').to_numpy()  # Essential to make sure all values are of one dtype
    model = hdbscan.HDBSCAN(min_samples=param_dict['min_samples'],
                            min_cluster_size=param_dict['min_cluster_size'],
                            cluster_selection_method=param_dict['cluster_selection_method'],
                            cluster_selection_epsilon=param_dict['epsilon'],
                            core_dist_n_jobs=n_cpus,
                            prediction_data=True)
    hdbscan_labels = model.fit_predict(X)
    with open(jsonFile, 'r') as j:
        param_dict = json.load(j)
    behavior_labels = param_dict['behavior_labels']
    for behavior, labels in behavior_labels.items():
        fig = show_labels(df, hdbscan_labels, important_labels = labels, add_text=True)
        fig.suptitle(behavior)
        fig.savefig(f'{saveDir}/final_{behavior}_labels.jpg')

if __name__ == '__main__':
    parser = ArgumentParser(description="Fit UMAP/HDBSCAN models to portion of the data and save, so that we can load those models any time we want to predict on other data")
    parser.add_argument('--embed_dataset', help='embedDataset .csv file from final_train_models.py', required=True)
    parser.add_argument('--param_file', help='Path to .txt file with UMAP and HDBSCAN parameters', default = 'fine_behaviors_analysis/models/more_test_params.txt')
    parser.add_argument('--output_dir', help='Location where behavior_labels.json and any figues will be saved', default = 'fine_behaviors_analysis/models')
    args = parser.parse_args()

    json_out = get_behavior_labels(args.embed_dataset,
                                   args.output_dir,
                                   args.param_file,
                                   show_figure=True,save_figure=False)

    show_final_clusters(embedDataset=args.embed_dataset,
                        saveDir=args.output_dir,
                        paramFile=args.param_file,
                        jsonFile=json_out)
    