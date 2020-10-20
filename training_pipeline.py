"""Training pipeline using pytorch."""

import torch
import torch.nn as nn

# 1) Design model architecture (inputs, outputs, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop (forward pass, backward pass, update weights)

# Linear regression
# f = w * x

# here : f = 2 * x

# Training samples
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

n_samples, n_features = X.shape
print(f"#samples: {n_samples}, #features: {n_features}")

X_test = torch.tensor([5], dtype=torch.float32)

# Design model
input_size = n_features
output_size = n_features
model = nn.Linear(input_size, output_size)

print(f"Prediction before training: f(5) = {model(X_test).item():.3f}")

# Define loss and optimizer
LR = 0.01
EPOCHS = 100
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    y_predicted = model(X)
    loss = criterion(Y, y_predicted)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f"epoch: {epoch + 1}, loss: {loss.item():.3f}")

print(f"Prediction after training: f(5) = {model(X_test).item():.3f}")
