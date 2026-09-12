import os
import pickle
import matplotlib.pyplot as plt
from utils import svd_3d, setup_mouse, prep_for_train
import umap
import numpy as np

b_map = {0: 0,
		 'rear': 1,
		 'face groom': 2,
		 'body groom': 3}
b_unmap = {value: key for key, value in b_map.items()}

def create_embedding(data_x, data_y, data_g, show = False):
	"""
	Fit UMAP on a portion of the data, then transform whole dataset
	"""

	n_neighbors = 35
	min_dist = 0.1
	n_components = 2
	spread = 1

	data_x_ = data_x[::10]
	data_y_ = data_y[::10]

	reducer = umap.UMAP(n_neighbors=n_neighbors,
						min_dist = min_dist,
						n_components = n_components,
						verbose = True,
						spread = spread,
						random_state = 1)
	umap_model = reducer.fit(data_x_)

	if show:
		fig, ax = plt.subplots(1,1)
		fig.suptitle(f"{n_neighbors=}, {min_dist=}")
		for b in np.unique(data_y_):
			values = umap_model.embedding_[data_y_ == b]
			ax.scatter(values[:, 0], values[:, 1], label = b_unmap[b], s=.8)
		ax.legend()
		plt.show()

	x_embed = umap_model.transform(data_x)
	return x_embed, data_y, data_g, umap_model

def _preprocess(path):
	fullpath, etho_path = setup_mouse(path)
	mouse = svd_3d(fullpath, etho_path)
	prep_for_train(mouse)

	data_x = mouse.distance_df
	if etho_path is not None:
		data_y = mouse.etho.map(lambda x: b_map[x])
	else:
		data_y = np.zeros(len(data_x))
	data_g = [os.path.split(mouse.single_id)[-1]] * len(data_x)

	return data_x, data_y, data_g

def load_dataset(path):
	assert path.endswith('.pickle'), 'path must be to a dataset .pickle file'
	with open(path, 'rb') as file:
		dataset = pickle.load(file)
	return dataset

def load_umap(path):
	assert path.endswith('.pickle'), 'path must be to a umap .pickle file'
	with open(path, 'rb') as file:
		umap_model = pickle.load(file)
	return umap_model

def preprocess(data_path, model_path = None):
	data_x, data_y, data_g = _preprocess(data_path)
	umap_model = load_umap(model_path)
	x_embed = umap_model.transform(data_x)
	return x_embed, data_y, data_g, umap_model

def save_dataset(x_embed, data_y, data_g, umap_model, dataset_name, output_path):
	data_path = os.path.join(output_path, dataset_name + '_dataset.pickle')
	with open(data_path, 'wb') as file:
		pickle.dump({'x_embed': x_embed,
					 'data_y' : data_y,
					 'data_g' : data_g}, file)

	model_path = os.path.join(output_path, dataset_name + '_umap.pickle')
	with open(model_path, 'wb') as file:
		pickle.dump(umap_model, file)



if __name__ == '__main__':
	print(f'Test')
