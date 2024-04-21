import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNMNIST(nn.Module):
    def __init__(self, network_config=dict()):
        super(FFNMNIST, self).__init__()
        self.hidden_dimensions = hidden_dimensions = network_config['hidden_dimensions']
        assert network_config['nonlinearity'] == 'relu', 'Only ReLU is supported as nonlinearity for now'
        self.layers = nn.ModuleDict() # for correctly registering these dynamically created layers
        self.layers['fc1'] = nn.Linear(784, hidden_dimensions[0])
        self.layers['act1'] = nn.ReLU()
        for i in range(1, len(hidden_dimensions)):
            self.layers[f'fc{i+1}'] = nn.Linear(hidden_dimensions[i-1], hidden_dimensions[i])
            self.layers[f'act{i+1}'] = nn.ReLU()
        self.layers['fc'] = nn.Linear(hidden_dimensions[-1], 10)

    def forward(self, x):
        out = x.view(-1, 784)
        for i in range(len(self.hidden_dimensions)):
            out = self.layers[f'fc{i+1}'](out)
            out = self.layers[f'act{i+1}'](out)
        out = self.layers['fc'](out)
        return out


def ffn_mnist(pretrained: bool = False, progress: bool = True, num_classes: int = 10, network_config=dict()):
    return FFNMNIST(network_config=network_config)