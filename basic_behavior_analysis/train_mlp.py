import os
import pandas as pd
import matplotlib.pyplot as plt
import torch

import numpy as np
import basic_behavior_analysis.preprocessing as pp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class PointsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
b_map = {0: 0,
            'rear': 1,
            'face groom': 2,
            'body groom': 3}
b_unmap = {value: key for key, value in b_map.items()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='basic_behavior_analysis/models/refined_initial_proj_data.pickle', help = '.pickle containing training dataset')
    parser.add_argument('--umap', default='basic_behavior_analysis/models/card9_gui_4.pickle', help='.pickle containing fitted umap model')
    parser.add_argument('--eval_data', default='data/example_mouse', help='Path to an anipose project folder')
    parser.add_argument('--output_path', help = '(Optional) File to save the trained MLP in', required=False)
    args = parser.parse_args()

    dataset = pp.load_dataset(args.dataset)
    umap_model = pp.load_umap(args.umap)

    data_x = dataset['x_embed']
    data_y = dataset['data_y']
    data_g = dataset['data_g']

    print(f'x: {len(data_x)}\ny: {len(data_y)}\ng: {len(data_g)}\n')
    fig, ax = plt.subplots(1,1)
    for b in np.unique(data_y):
        values = data_x[data_y == b]
        ax.scatter(values[:, 0], values[:, 1], label = b_unmap[b], s=.8)
    ax.legend()
    plt.show()

    # Create dataset and dataloaders
    X, y = data_x, data_y
    dataset = PointsDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True) #, num_workers = 8, pin_memory = True) # Appx 2:20 per epoch with 8 workers

    # Model, loss function, and optimizer
    input_size = 2  # Input is 2D coordinates
    hidden_size = 64  # Number of hidden units in the MLP
    num_classes = 4  # Number of categories in dataset (behavior labels)

    model = MLP(input_size, hidden_size, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device=}')
    model.to(device)

    class_counts = np.bincount(y)
    total_count = len(y)
    class_weights = total_count / (len(np.unique(y)) * class_counts)
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print(f'These are the weights: {weights}')
    # CrossEntropyLoss with class weights
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Training loop
    num_epochs = 50  # Number of training epochs

    losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            # Move data to the same device as the model (GPU or CPU)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    plt.plot(losses)

    # 4. Evaluate the model (you can add test data here)
    x_embed, holdout_y, holdout_g, _ = pp.preprocess(args.eval_data,
                                                     model_path=args.umap)

    model.eval()
    with torch.no_grad():
        test_points = torch.tensor(x_embed)  # Example points to classify
        test_points = test_points.to(device)  # Move test points to GPU
        outputs = model(test_points)
        _, predicted = torch.max(outputs, 1)
        predictions_df = pd.DataFrame(predicted.cpu().numpy())

    color_dict = {0: 'black',
                  1: 'blue',
                  2: 'red',
                  3: 'green'}
    b_b_map = {0: 'None',
               1: 'Rear',
               2: 'Face Groom',
               3: 'Body Groom'}
    fig = plt.figure()
    embed_df = pd.DataFrame(x_embed)
    for key, value in b_b_map.items():
        print(key)
        if (predictions_df == key).sum().values > 0:
            beh_embed = embed_df.loc[(predictions_df == key).values]
            plt.scatter(beh_embed.iloc[:, 0], beh_embed.iloc[:, 1], color = color_dict[key], label = value, s = .5)
    plt.legend()
    plt.show()

    # Optionally, visualize the data
    plt.scatter(x_embed[:, 0], x_embed[:, 1], c=predicted.cpu().numpy(), cmap=plt.cm.jet, s = .5)

    print(predictions_df.value_counts())
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("2D Dataset")
    plt.show()

    #torch.save(model.state_dict(), args.output_path)
    
