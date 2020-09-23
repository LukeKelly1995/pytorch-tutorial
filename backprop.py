"""Backpropagation basics in pytorch
"""

import torch

x = torch.tensor(1.0)
print(f"Input value: {x}")
y = torch.tensor(2.0)
print(f"Target value: {y}")

w = torch.tensor(1.0, requires_grad=True)
print(f"Initialised weight value: {w}")
print()

print("Forward pass...")
y_predicted = w * x
print(f"Predicted target (w * x): {y_predicted}")
loss = (y_predicted - y) ** 2
print(f"Squared error ((y_predicted - y) ** 2): {loss}")
print()

print("Backwards pass...")
loss.backward()
print(
    f"Gradient of Loss with respect to the weights (dLoss/dw) via the chain rule: {w.grad}"
)
print()

print("Optimisation step...")
LEARNING_RATE = 0.01
with torch.no_grad():
    print(f"Old weight: {w}")
    print(f"Weight modification: {LEARNING_RATE * w.grad}")
    w -= LEARNING_RATE * w.grad
    print(f"Updated weight: {w}")

print("Zero gradients after optimisation step")
w.grad.zero_()
print()

print("Second forward pass to check loss has decreased...")
y_predicted = w * x
print(f"Predicted target (w * x): {y_predicted}")
loss = (y_predicted - y) ** 2
print(f"Squared error ((y_predicted - y) ** 2): {loss}")
print()
