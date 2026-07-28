import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, h_dim, r_dim):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(2, h_dim)
        self.linear2 = nn.Linear(h_dim, h_dim)
        self.linear3 = nn.Linear(h_dim, r_dim)
        self.relu = nn.ReLU()

    def forward(self, x_o, y_o):
        # Concatenate each observed (x_o, y_o) pair
        inputs = torch.cat([x_o, y_o], dim=-1)
        r = self.relu(self.linear1(inputs))
        r = self.relu(self.linear2(r))
        r = self.linear3(r)
        # Aggregate by averaging over all observed points
        r_O = r.mean(dim=1)
        return r_O

class Decoder(nn.Module):
    def __init__(self, r_dim, h_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(r_dim + 1, h_dim)
        self.linear2 = nn.Linear(h_dim, h_dim)
        self.linear3 = nn.Linear(h_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, r_O, x_t):
        # Concatenate r_O with each x_t
        x_t = x_t.reshape(x_t.shape[0], x_t.shape[1], 1)
        r_O = r_O.reshape(r_O.shape[0], 1, r_O.shape[1]).repeat(1, x_t.shape[1], 1)
        inputs = torch.cat([r_O, x_t], dim=-1)  
        h = self.relu(self.linear1(inputs))
        h = self.relu(self.linear2(h))
        y_hat = self.linear3(h)
        return y_hat

# Define the full model that combines the encoder and decoder
class Model(nn.Module):
    def __init__(self, h_dim, r_dim):
        super(Model, self).__init__()
        self.encoder = Encoder(h_dim, r_dim)
        self.decoder = Decoder(r_dim, h_dim)

    def forward(self, x_o, y_o, x_t):
        r_O = self.encoder(x_o, y_o)
        y_hat = self.decoder(r_O, x_t)
        return y_hat
