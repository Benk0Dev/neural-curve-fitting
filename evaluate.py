import torch
from dataset import x_test, y_test

def evaluate(model, test_contexts, loss_fn):
    model.eval() # Set the model to evaluation mode
    with torch.no_grad():
        for key, context in test_contexts.items():
            x_o = context['x']
            y_o = context['y']

            # Forward pass
            y_hat = model(x_o, y_o, x_test)

            # Compute average MSE across all test functions
            loss = loss_fn(y_hat, y_test.reshape(y_test.shape[0], y_test.shape[1], 1))
            
            print(f"  {key}, MSE: {loss.item():.6f}")
    model.train() # Set the model back to training mode
