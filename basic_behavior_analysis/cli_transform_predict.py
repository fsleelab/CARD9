import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import basic_behavior_analysis.preprocessing as pp
from basic_behavior_analysis.train_mlp import MLP
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

b_map = {0: 0,
         'rear': 1,
         'face groom': 2,
         'body groom': 3}

b_unmap = {value: key for key, value in b_map.items()}

def transform(date_dirs, output_dir, save_figure, mlp_path, umap_path):
    model = MLP(input_size = 2, hidden_size = 64, num_classes = 4)
    model.load_state_dict(torch.load(mlp_path, weights_only=True, map_location=device))
    model.to(device)

    if not type(date_dirs) is list:
        date_dirs = [date_dirs]
    temp = []
    for dir_ in date_dirs:
        content = [i for i in os.listdir(dir_)]
        content = [os.path.join(dir_, i) for i in content]
        temp.append(content)
    all_dirs = [item for sub in temp for item in sub]
    all_dirs = [i for i in all_dirs if (os.path.isdir(i) and not 'temp_project' in i)]
    all_dirs = [i for i in all_dirs if (os.path.isdir(os.path.join(i, 'pose-3d')))]


    all_embeddings = []
    all_classifications = []
    names = []
    for subdir in all_dirs:
        name = os.path.split(subdir)[-1]
        
        x_embed, holdout_y, holdout_g, _ = pp.preprocess(subdir,
                                                         model_path=umap_path)
        names.append(holdout_g)
        all_embeddings.append(x_embed)
        with torch.no_grad():
            test_points = torch.tensor(x_embed)  # Example points to classify
            test_points = test_points.to(device)  # Move test points to GPU
            outputs = model(test_points)
            _, predicted = torch.max(outputs, 1)
            predictions_df = pd.DataFrame(predicted.cpu().numpy())
            all_classifications.append(predictions_df)
        results = predictions_df.value_counts()
        mapped_df = []
        for count, value in enumerate(results):
            count = b_unmap[results.index[count][0]]
            mapped_df.append((count, value))
        mapped_df = pd.DataFrame(mapped_df)
        savepath = os.path.join(output_dir, name + '.csv')
        mapped_df.to_csv(savepath, index = None, header = None)
        print(f'Printed to {savepath}')

    if save_figure:
        embeddings = np.concatenate(all_embeddings)
        class_df = pd.concat(all_classifications, axis = 0)
        names = np.concatenate(names)
        embed_df = pd.DataFrame(embeddings)

        rotation_mtx = np.array([[0, 1],
                                 [-1, 0]])
        centroid = np.mean(embed_df.values, axis = 0)
        translated_points = embed_df - centroid
        rotated_translated = translated_points @ rotation_mtx.T
        rotated_points = rotated_translated + centroid
        embed_df = pd.DataFrame(rotated_points)

        curr_names = [np.array(i) for i in np.unique(names)]
        fig, axs = plt.subplots(1,1)
        for count, name in enumerate(curr_names):
            comp_x = embed_df.values[names == name]
            comp_y = class_df.values[names == name].squeeze()
            for b in np.unique(comp_y):
                values = comp_x[comp_y == b]
                axs.scatter(values[:, 0], values[:, 1], label=b_unmap[b], s=.25)

        fig.legend(fontsize='medium', markerscale = 10)
        plt.savefig(os.path.join(output_dir, 'summary_figure' + '.jpg'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', help='Path to project folders used for input', default='data')
    parser.add_argument('--output_dir', help='Location where predictions will be saved to', default='output')
    parser.add_argument('--savefig', action='store_true', help='Set to save a summary figure in output_dir showing predicted labels')
    parser.add_argument('--mlp_path', help='Path to trained MLP .pickle', default = 'basic_behavior_analysis/models/temp_save.pth')
    parser.add_argument('--umap_path', help='Path to fitted UMAP reducer .pickle', default = 'basic_behavior_analysis/models/card9_gui_4.pickle')
    args = parser.parse_args()
    
    transform(args.project, args.output_dir, args.savefig, args.mlp_path, args.umap_path)

