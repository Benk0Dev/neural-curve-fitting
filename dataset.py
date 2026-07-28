import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

y_train = np.loadtxt('data/train_data.csv', delimiter=',')
N_train, Nx = y_train.shape
x_train = np.tile(np.linspace(-1, 1, Nx), (N_train, 1))
x_tensor = torch.from_numpy(x_train).float()
y_tensor = torch.from_numpy(y_train).float()
train_loader = DataLoader(TensorDataset(x_tensor, y_tensor),batch_size=128,shuffle=True)

y_test = torch.from_numpy(np.loadtxt('data/test_data.csv', delimiter=',')).float()
x_test = torch.linspace(-1, 1, 100).unsqueeze(0).repeat(300, 1)

# Load the context bundle
test_contexts = torch.load('data/test_observations.pt')
    