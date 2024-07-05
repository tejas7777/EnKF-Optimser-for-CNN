import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=5):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 32  # Reduced the number of filters to make the model smaller
        
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        y = self.dncnn(x)
        return x - y  # Residual learning