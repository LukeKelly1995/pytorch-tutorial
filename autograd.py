"""Autograd basics in pytorch
"""

import torch

# requires_grad = True -> tracks all operations on the tensor
x = torch.randn(3, requires_grad=True)
y = x + 2

# y was created as a result of an operation, so it has a grad_fn attribute
# grad_fn: references a Function that has created the Tensor
print(f"Random tensor created by user: {x}")  # create by the user -> grad_fn is None
print(f"Tensor created by operation on existing tensor: {y}")
print(f"Gradient function of y tensor: {y.grad_fn}")
print()

# do more operations on y
z = y * y * 3
z = z.mean()
z.backward()  # dz/dx
print(f"Gradient of x tensor with scalar output: {x.grad}")
print()

# model with non-scalar output:
# if a Tensor is non-scalar (more than 1 elements), we need to specify arguments for backward()
# specify a gradient argument that is a tensor of matching shape.
# needed for vector-Jacobian product

x = torch.randn(3, requires_grad=True)
y = x * 2
for _ in range(10):
    y = y * 2

print(f"Tensor created by operation on existing tensor: {y}")
print(f"Tensor shape: {y.shape}")
print()

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
y.backward(v)
print(f"Gradient of x tensor with non-scalar output: {x.grad}")
print()

# stop a tensor from tracking history
# 1) x.requires_grad_(False)
a = torch.randn(2, 2)
print(f"Tensor a requires gradient: {a.requires_grad}")
b = (a * 3) / (a - 1)
print(f"Tensor b  gradient function: {b.grad_fn}")
a.requires_grad_(True)
print(f"Tensor a requires gradient: {a.requires_grad}")
b = (a * a).sum()
print(f"Tensor b gradient function: {b.grad_fn}")
print()

# 2) x.detach()
a = torch.randn(2, 2, requires_grad=True)
print(f"Tensor a requires gradient: {a.requires_grad}")
b = a.detach()
print(f"Tensor b requires gradient: {b.requires_grad}")
print()

# 3) wrap in with torch.no_grad(): context manager
a = torch.randn(2, 2, requires_grad=True)
print(f"Tensor a requires gradient: {a.requires_grad}")
with torch.no_grad():
    b = x ** 2
    print(f"Tensor b requires gradient: {b.requires_grad}")
    print()

# empty gradients
weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    # fake a training loop
    model_output = (weights * 3).sum()
    model_output.backward()

    print(f"Gradients of weights tensor: {weights.grad}")

    # opitmise the model
    with torch.no_grad():
        weights -= 0.1 * weights.grad

    # flush out the gradients
    weights.grad.zero_()

print(f"Weights tensor: {weights}")
print(f"Model output: {model_output}")
print()

# optimisers have zero_grad() method
optimizer = torch.optim.SGD([weights], lr=0.1)
# during training:
optimizer.step()
optimizer.zero_grad()
