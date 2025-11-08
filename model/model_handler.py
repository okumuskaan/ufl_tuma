"""Model factory and simple architectures (MLP for FMNIST, CNN for CIFAR)."""
import torch.nn as nn 
import torch.nn.functional as F 
import math

def modelHandler(architecture, input_shape, output_dim):
    if architecture == "cnn":
        return CNN(num_classes=output_dim)
    elif architecture == "MLP":
        input_dim = int(math.prod(input_shape))
        return MLP_FMNIST(dim_in=input_dim, dim_hidden1=64, dim_hidden2=30, dim_out=output_dim)
    else:
        raise ValueError(f"Unknown architecture {architecture}")        

class MLP_FMNIST(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, dim_out):
        super(MLP_FMNIST, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden1 = nn.Linear(dim_hidden1, dim_hidden2)
        self.layer_hidden2 = nn.Linear(dim_hidden2, dim_out)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden2(x)

        return self.logsoftmax(x)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, 84)
        self.fc4 = nn.Linear(84, 50)
        self.fc5 = nn.Linear(50, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return self.logsoftmax(x)