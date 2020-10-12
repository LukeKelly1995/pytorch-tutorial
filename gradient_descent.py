"""Backpropagation using autograd in pytorch."""

import numpy as np
import torch

# Linear regression computing every step manually
# f = w * x
# here: f = 2 * x

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

W = 0.0

# model output
def forward(x_input):
    """Emulates forward pass of model.

    Args:
        x_input ([float32]): model input

    Returns:
        [float32]: model output
    """
    return W * x_input


# loss = Mean Square Error
def loss(y_actual, y_pred):
    """Mean squared error regression loss function.

    Args:
        y_actual ([float32]): target's actual value
        y_pred ([float32]): target's predicted value

    Returns:
        [float32]: mean squared error of predictions
    """
    return ((y_pred - y_actual) ** 2).mean()


# J = MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x(w*x - y)
def gradient(x_input, y_actual, y_pred):
    """Computes the gradient with respect to the weight.

    Args:
        x_input ([float32]): model input
        y_actual ([float32]): target's actual value
        y_pred ([float32]): target's predicted value

    Returns:
        [float32]: [description]
    """
    return np.dot(2 * x_input, y_pred - y_actual).mean()


print(f"Prediction before training: f(5) = {forward(5):.3f}")

# Training parameters
LR = 0.01
EPOCHS = 20

for epoch in range(EPOCHS):
    # predict = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # calculate gradients
    dw = gradient(X, Y, y_pred)

    # update weights
    W -= LR * dw

    if epoch % 2 == 0:
        print(f"epoch {epoch+1}: w = {W:.3f}, loss = {l:.8f}")
print(f"Prediction after training: f(5) = {forward(5):.3f}")

# Here we replace the manually computed gradient with autograd

# Linear regression
# f = w * x

# here : f = 2 * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

W = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

for epoch in range(EPOCHS):
    # predict = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # calculate gradients = backward pass
    l.backward()

    # update weights
    # w.data = W.data - learning_rate * W.grad
    with torch.no_grad():
        W -= LR * W.grad

    # zero the gradients after updating
    W.grad.zero_()

    if epoch % 10 == 0:
        print(f"epoch {epoch+1}: W = {W.item():.3f}, loss = {l.item():.8f}")

print(f"Prediction after training: f(5) = {forward(5).item():.3f}")
