import numpy as np
import torch

from dataset import train_loader, test_contexts
from model import Model
from train import train, h_dim, num_epochs, lr, n_min, n_max, loss_fn
from evaluate import evaluate

r_dims = [2, 4, 8]

models = {}
for r_dim in r_dims:
    print(f"Training model{r_dim}:")
    model = Model(h_dim=h_dim, r_dim=r_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train(model, train_loader, optimizer, loss_fn, num_epochs, n_min, n_max)
    models[r_dim] = model
    print()

for r_dim, model in models.items():
    print(f"Evaluating model{r_dim}:")
    evaluate(model, test_contexts, loss_fn)
    print()


model4 = models[4]
model4.eval() # Set the model to evaluation mode

context = test_contexts['20_even']
x_o = context['x']
y_o = context['y']

with torch.no_grad():
    # Pass context points through the encoder to get latent representations
    r_O = model4.encoder(x_o, y_o)

r_O_numpy = r_O.numpy()
# Transpose so each row represents one dimension across all 300 functions
correlation_matrix = np.corrcoef(r_O_numpy.T)

print("Correlation matrix:")
print(np.round(correlation_matrix, 3))
