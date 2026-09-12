import os
from utils import svd_3d, prep_for_train, roll_up
import numpy as np
import umap
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import hdbscan
from fine_behaviors_analysis.txt_reader import read_txt
import itertools
from argparse import ArgumentParser

class revised_mouse(svd_3d):
    """
    Essentially identical to svd_3d except when reading BORIS hand scoring data, which we make more efficient
    and adapt to work with the behaviors we are including as part of the revision
    """
    def __iter__(self, file, etho_file, manual_z_flip = None, show = False):
        super().__init__(file, etho_file, manual_z_flip, show)
    def align_etho(self, FPS = 30):
        if self.etho_file is None:
            frame_ethos = pd.Series(index=range(len(self.rotated_df)))
            frame_ethos[0:len(self.rotated_df)] = 0
            return frame_ethos
        etho = pd.read_csv(self.etho_file, header=15)
        if 'Behavior' not in etho.columns:  # Such is the case if you used two videos or more to score
            for i in range(1, 4):
                etho = pd.read_csv(self.etho_file, header=15 + i)
                if 'Behavior' in etho.columns:
                    break
        if not all(etho['Status'][::2] == 'START') and all(etho['Status'][1::2] == 'STOP'):
            raise IndexError('Etho file error: START/STOP does not alternate properly')
        if not all(x == z for x, z in zip(etho['Media file path'][::2], etho['Media file path'][1::2])):
            raise IndexError('Etho file error: Only one type of behavior can be STARTed at a time')
        etho_data = np.zeros(len(self.rotated_df), dtype='U15')

        # Replace face-grooming specific terms with just face groom
        behaviors = etho['Behavior'].unique()
        unusedLabels = ['bilateral', 'elliptical', 'unilateral']
        if any(i in behaviors for i in unusedLabels):
            etho['Behavior'] = etho['Behavior'].replace({i: 'face groom' for i in unusedLabels})

        # Aesthetic choice to rename some labels
        replaceTerms = {
            'biting rear leg' : 'body groom',
            'scratch w hind leg' : 'hindpawScratch'
        }
        if any(i in behaviors for i in replaceTerms.keys()):
            etho['Behavior'] = etho['Behavior'].replace(replaceTerms)
        behaviors = etho['Behavior'].unique()

        priorityOrder = [ # This ensures that double-labels (e.g. scratch inside of a body groom period)
            'rear',       # result in the most important label being represented in the final etho dataframe.
            'body groom',
            'face groom',
            'hindpawGroom',
            'hindpawScratch',
            'twitch',
        ]
        for beh in priorityOrder:
            if beh not in behaviors:
                continue
            if beh == 'twitch': # Point event: have to use a single duration per event
                behMask = etho['Behavior'] == 'twitch'
                starts = etho[behMask]['Time'].values
                stops = starts + (10 / FPS) # Twitches seem to take at most 10 frames to complete, but may need to tweak this
            else:
                behMask = etho['Behavior'] == beh
                starts = etho[behMask]['Status'] == 'START'
                starts = etho[behMask]['Time'][starts].values
                stops = etho[behMask]['Status'] == 'STOP'
                stops = etho[behMask]['Time'][stops].values
            for on, off in zip(starts,stops):
                #print(f'Segment {on} - {off}')
                if int(off * FPS) > len(etho_data) - 1:
                    etho_data[int(on * FPS):] = beh
                else:
                    etho_data[int(on * FPS):int(off * FPS)] = beh
        etho_data = pd.Series(etho_data)
        msk = etho_data == ''
        etho_data.loc[msk] = 0
        return etho_data

def longlistdir(dir):
    dirs = os.listdir(dir)
    dirs = [os.path.join(dir, i) for i in dirs if not i.startswith('.')]
    return dirs

def load_etho_mice(allDataDir):
    mice = []
    for dir in longlistdir(allDataDir):
        ethoFile = os.path.join(allDataDir, dir, 'revised_etho.csv')
        if not os.path.isfile(ethoFile):
            continue
        poseFile = [i for i in longlistdir(f'{dir}/pose-3d') if i.endswith('.csv')][0]
        mouse = revised_mouse(poseFile, ethoFile, show=False)
        prep_for_train(mouse)
        mice.append(mouse)
    return mice

def preprocess_data(mice, param_dict, allowPrune):
    """
    Prepare arrays that will be fit/transformed by UMAP.
    allowPrune should be set to false when transforming unlabeled mice (i.e. during inference).
    """
    data_roll = []
    etho_roll = []
    for mouse in mice:
        d, e = roll_up(mouse,
                       period=param_dict['period'],
                       dataframe=param_dict['df_choice'],
                       save_space=False,
                       restrict_bps=param_dict['restrict_bps']
                       )
        data_roll.append(d)
        etho_roll.append(e)
    data_roll = np.vstack(data_roll)
    etho_roll = pd.concat(etho_roll, axis=0).reset_index(drop=True)
    if not allowPrune:
        return data_roll, etho_roll

    # During training/parameter search, we want to downsize the dataset size without losing data
    # containing rare behaviors, so we exempt those data points from the pruning process
    reserved = etho_roll[etho_roll['etho'].isin(['twitch', 'hindpawGroom', 'hindpawScratch'])]
    prunable = etho_roll[~etho_roll['etho'].isin(['twitch', 'hindpawGroom', 'hindpawScratch'])]
    pruned = prunable[::param_dict['prune_by']]

    # combine and sort
    final_indices = np.sort(np.concatenate([
        reserved.index,
        pruned.index
    ]))
    data_roll = data_roll[final_indices]
    etho_roll = etho_roll.loc[final_indices].reset_index(drop=True)
    return data_roll, etho_roll

def fit_transform(data_roll, etho_roll, param_dict):
    """
    Fit a UMAP model on the data conditioned with labels
    """
    y = etho_roll['etho'].values
    etho_map = {}
    for count, i in enumerate(etho_roll['etho'].unique()):
        if i == 0 or i == -1:
            y[y == i] = -1
            etho_map[-1] = i
        else:
            y[y == i] = count
            etho_map[count] = i

    reducer = umap.UMAP(n_neighbors=param_dict['n_neighbors'],
                        min_dist=param_dict['min_dist'],
                        n_components=param_dict['n_components'],
                        verbose=True,
                        spread=param_dict['spread'],
                        random_state=1,
                        target_metric='categorical')
    embedding = reducer.fit_transform(data_roll, y=y)
    embed_df = pd.DataFrame(embedding)
    etho_roll['etho'] = etho_roll['etho'].map(etho_map)
    return embed_df, reducer

def load_unlabeled_mice(allDataDir, numToLoad = 10):
    unlabeled = []
    count = 0
    for dir in longlistdir(allDataDir):
        ethoFile = os.path.join(allDataDir, dir, 'revised_etho.csv')
        if os.path.isfile(ethoFile) or not os.path.isdir(os.path.join(allDataDir, dir)):
            continue
        if count == numToLoad:
            break
        poseFile = [i for i in longlistdir(f'{dir}/pose-3d') if i.endswith('.csv') and 'frame_synced' in i][0]
        mouse = revised_mouse(poseFile, None, show=False)
        prep_for_train(mouse)
        #mouse.distance_df = mouse.distance_df.astype(np.float32)
        unlabeled.append(mouse)
        count += 1
    return unlabeled

def iter_unlabeled_mice(allDataDir):
    """
    Generator that yields one 'mouse' at a time, remembering its position
    as long as you keep using the same generator instance.
    """
    for d in longlistdir(allDataDir):
        ethoFile = os.path.join(allDataDir, d, 'revised_etho.csv')
        if os.path.isfile(ethoFile):
            continue

        pose_dir = os.path.join(allDataDir, d, 'pose-3d')
        pose_files = [f for f in longlistdir(pose_dir) if f.endswith('.csv') and 'frame_synced' in f]
        if not pose_files:
            continue

        poseFile = pose_files[0]  # or whatever selection rule you want
        mouse = revised_mouse(poseFile, None, show=False)
        prep_for_train(mouse)

        yield mouse

def expand_dataset(allDataDir, txtFile, saveDir, save_model = True):
    """
    After fitting and saving the UMAP model, transform data not in training set and combine to create a larger dataset.
    This expanded dataset is what HDBSCAN will fit on.
    """
    def _load_unlabeled_mice(numToLoad=10):
        """
        Uses the persistent generator to get the next N mice.
        Each call continues from where the previous one left off.
        """
        return list(itertools.islice(_unlabeled_gen, numToLoad))
    _unlabeled_gen = iter_unlabeled_mice(allDataDir)

    param_dict = read_txt(txtFile)
    param_dict['df_choice'] = 'distance'
    param_dict['n_components'] = 2
    print(f'This is param_dict: {param_dict}')

    mice = load_etho_mice(allDataDir)
    if param_dict['num_mice'] > 16:
        addmice =_load_unlabeled_mice(param_dict['num_mice'] - 16)
        mice = mice + addmice

    data_roll, etho_roll = preprocess_data(mice, param_dict, True)
    embed_df, reducer = fit_transform(data_roll, etho_roll, param_dict)

    saveName = os.path.split(txtFile)[-1].split('.')[0]

    embed_df.to_csv(os.path.join(saveDir, saveName + '_embed_df.csv'))
    etho_roll.to_csv(os.path.join(saveDir, saveName + '_etho_roll.csv'))

    if save_model:
        umap_save_path = os.path.join(saveDir, saveName + '_model.sav')
        assert not os.path.isfile(umap_save_path), f'UMAP reducer already saved to {umap_save_path}'
        with open(umap_save_path, 'wb') as f:
            pickle.dump(reducer, f)

    fig = create_figure(embed_df, etho_roll, 'labeled_data_only')
    savefigname = os.path.join(saveDir, f'{saveName}_labeled_data.jpg')
    fig.savefig(savefigname)
    plt.close(fig)
    del embed_df, data_roll, etho_roll, mice # Clear up memory space

    mice = _load_unlabeled_mice(20)
    data_roll, etho_roll = preprocess_data(mice, param_dict, False)
    embedding = reducer.transform(data_roll)
    embed_df = pd.DataFrame(embedding)

    _embed_df = pd.read_csv(os.path.join(saveDir, saveName + '_embed_df.csv'), index_col=0, header=0)
    _etho_roll = pd.read_csv(os.path.join(saveDir, saveName + '_etho_roll.csv'), index_col=0, header=0)

    etho_roll_col = etho_roll.columns
    full_embed_df = pd.DataFrame(np.vstack([embed_df.values, _embed_df.values]))
    full_etho_roll = pd.DataFrame(np.vstack([etho_roll.values, _etho_roll.values]))
    full_etho_roll.columns = etho_roll_col

    full_embed_df.to_csv(os.path.join(saveDir, saveName + '_full_embed.csv'))
    full_etho_roll.to_csv(os.path.join(saveDir, saveName + '_full_etho.csv'))
    combined_for_clustering = pd.concat([full_embed_df, full_etho_roll['etho']], axis=1)
    combined_for_clustering.columns = ['x','y','etho']
    embedDataset = os.path.join(saveDir, saveName + '_to_cluster.csv')
    combined_for_clustering.to_csv(embedDataset)

    fig = create_figure(full_embed_df, full_etho_roll, 'full_dataset')
    savefigname = os.path.join(saveDir, f'{saveName}_full_dataset.jpg')
    fig.savefig(savefigname)
    plt.close(fig)
    return embedDataset

def fit_HDBSCAN(embedDataset,
             saveDir,
             paramFile,
             show_figure=False,
            save_figure=False,
             save_model=False,
             n_cpus=1):

    param_dict = read_txt(paramFile)
    saveName = os.path.split(paramFile)[-1].split('.')[0]

    df = pd.read_csv(embedDataset)
    
    X = df[['x', 'y']].to_numpy()
    y = df['etho'].to_numpy()

    model = hdbscan.HDBSCAN(min_samples=param_dict['min_samples'],
                            min_cluster_size=param_dict['min_cluster_size'],
                            cluster_selection_method=param_dict['cluster_selection_method'],
                            cluster_selection_epsilon=param_dict['epsilon'],
                            core_dist_n_jobs=n_cpus,
                            prediction_data=True).fit(X)

    if save_model:
        with open(os.path.join(saveDir, f'{saveName}_hdbscan_model.sav'), 'wb') as handle:
            pickle.dump(model, handle)
    if show_figure or save_figure:
        import matplotlib
        hdbscan_labels = model.fit_predict(X)
        fig = show_cluster_figure(X, hdbscan_labels, important_labels = None, add_text = True)
        if save_figure:
            fig.savefig(os.path.join(saveDir, f'{saveName}_labels.jpg'))
            plt.close(fig)
        colors = matplotlib.colors.ListedColormap(
            ['black', 'red', 'pink', 'blue', 'coral', 'yellow', 'green', 'purple', 'navy', 'black', 'red', 'pink',
             'blue', 'coral', 'yellow', 'green', 'purple'])
        color_dict = {}
        clustered_df = pd.DataFrame(hdbscan_labels, columns=['cluster_label'])
        fig, ax = plt.subplots(1, 1, figsize=(5, 7), dpi=150)
        for count, cluster_id in enumerate(clustered_df['cluster_label'].value_counts().index):
            data_in_cluster = X[clustered_df['cluster_label'] == cluster_id]
            if cluster_id == -1:
                color = (0.5, 0.5, 0.5, 0.5)  # Grey
                color_dict[cluster_id] = color
            else:
                color = colors(count)
                color_dict[cluster_id] = colors(count)
            ax.scatter(data_in_cluster[:, 0],
                       data_in_cluster[:, 1],
                       s=0.1,
                       color=color)
        ax.axis('off')
        if save_figure:
            fig.savefig(os.path.join(saveDir, f'{saveName}_clusters.jpg'))
            plt.close(fig)
        else:
            plt.show()

def create_figure(embed_df, etho_roll, globalStep):
    colorDict = {
        'rear': 'blue',
        'body groom': 'green',
        'face groom': 'lime',
        'hindpawGroom': 'orange',
        'hindpawScratch': 'red',
        'twitch': 'purple',
        0 : 'black',
        '0' : 'black',
        '0.0':'black',
    }

    fig = plt.figure(figsize=(6, 9), dpi=175)
    fig.suptitle(globalStep)

    ax = fig.add_subplot(111)
    ax.scatter(embed_df[0], embed_df[1], color='black', alpha=.007, s=0.5)
    for etho_type in etho_roll['etho'].unique():
        if type(etho_type) is not str:
            continue
        if etho_type not in ['hindpawScratch', 'hindpawGroom', 'twitch']:
            thinOut = 5
        else:
            thinOut = 1
        ax.scatter(embed_df[0][etho_roll['etho'] == etho_type][::thinOut],
                   embed_df[1][etho_roll['etho'] == etho_type][::thinOut],
                   s=0.5,
                   color=colorDict[etho_type],
                   label=etho_type,
                   edgecolors=None, alpha = 0.2)
    ax.legend(markerscale=8)
    return fig

def show_cluster_figure(X, hdbscan_labels, important_labels = None, add_text = False):
    import matplotlib as mpl
    all_labels = np.unique(hdbscan_labels)
    # if important_labels is a list, we make sure those get colored a bit differently than any other label
    if important_labels is not None:
        cmap = mpl.colormaps['winter'](np.linspace(0,1,len(all_labels)))
        cmap[np.isin(all_labels, important_labels)] = mpl.colormaps['autumn'](np.linspace(0,1,len(important_labels)))
    else:
        cmap = mpl.colormaps['hsv'](np.linspace(0,1,len(all_labels)))
    fig, ax = plt.subplots(1, 1, figsize=(6, 9), dpi=170)
    for count, cluster_id in enumerate(all_labels):
        data_in_cluster = X[hdbscan_labels == cluster_id]
        centroid = data_in_cluster.mean(axis=0)
        if cluster_id == -1:
            color = (0.5, 0.5, 0.5, 0.5)  # Grey
            text = None
        else:
            color = cmap[count]
            text = str(cluster_id)
        ax.scatter(data_in_cluster[:, 0],
                   data_in_cluster[:, 1],
                   s=0.1,
                   color=color)
        if add_text:
            ax.text(centroid[0], centroid[1], text, color = 'black', fontsize=8)
    ax.axis('off')
    return fig

if __name__ == "__main__":
    parser = ArgumentParser(description="Fit UMAP/HDBSCAN models to portion of the data and save, so that we can load those models any time we want to predict on other data")
    parser.add_argument('--labeled_data_dir', help='All annotated and unannotated project folders', required=True)
    parser.add_argument('--param_file', help='Path to .txt file with UMAP and HDBSCAN parameters', default = 'fine_behaviors_analysis/models/more_test_params.txt')
    parser.add_argument('--output_dir', help='Location where models and datasets will be saved', default = 'fine_behaviors_analysis/models')
    parser.add_argument('--save_model', action='store_true', help='Set to save models')
    args = parser.parse_args()

    embedDataset = expand_dataset(args.labeled_data_dir, args.param_file, args.output_dir, args.save_model)
    print(f'embedDataset saved at {embedDataset}')
    fit_HDBSCAN(embedDataset, args.output_dir, args.param_file, show_figure=True, save_figure=False, save_model=args.save_model)
