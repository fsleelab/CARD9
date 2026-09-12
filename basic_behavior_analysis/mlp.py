from torch import nn
import torch.nn.functional as F

class MLP_Classifier(nn.Module):
    def __init__(self,
                 layers_num,
                 device,
                 dropout,
                 hidden_channels = 4,
                 in_channels = 2,
                 out_channels = 4):
        super().__init__()
        self.device = device
        layers = self.get_layers(layers_num,
                                 in_channels,
                                 out_channels,
                                 hidden_channels)
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.c_loss_fn = nn.CrossEntropyLoss()


    def get_layers(self, layers_num, in_channels, out_channels, hidden_channels):
        layer_in = nn.Linear(in_channels, hidden_channels).to(self.device)
        layer_out = nn.Linear(hidden_channels, out_channels).to(self.device)
        layers_hidden = [nn.Linear(hidden_channels, hidden_channels).to(self.device) for i in range(layers_num)]

        layers = [[layer_in], layers_hidden, [layer_out]]
        layers = [item for subv in layers for item in subv]
        return layers

    def forward(self, x):
        for count, layer in enumerate(self.layers):
            if count == 0:
                #x_norm = F.normalize(x)
                x = layer(x)
                x = F.relu(x)
                x = self.dropout(x)
                continue
            if count == len(self.layers) - 1:
                break
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x