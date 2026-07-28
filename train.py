import torch
from torch import nn

def train(model, train_loader, optimizer, loss_fn, num_epochs, n_min, n_max):
    for epoch in range(num_epochs):
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            
            # Sample N random context points from each function in the batch
            N = torch.randint(n_min, n_max + 1, (1,)).item()
            idx = torch.randperm(x_batch.shape[1])[:N]
            x_o = x_batch[:, idx].reshape(x_batch.shape[0], N, 1)
            y_o = y_batch[:, idx].reshape(y_batch.shape[0], N, 1)

            # Forward pass
            y_hat = model(x_o, y_o, x_batch)

            # Compute loss against all target points
            loss = loss_fn(y_hat, y_batch.reshape(y_batch.shape[0], x_batch.shape[1], 1))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"  Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.6f}")
        
# Hyperparameters
h_dim = 128
num_epochs = 40
lr = 1e-3
n_min = 3
n_max = 20
loss_fn = nn.MSELoss()
