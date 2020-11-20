"""Tensor basics in pytorch."""

import torch

# torch.empty(size): uninitiallised tensor
scalar = torch.empty(1)
print(f"Uninitiallised scalar: \n{scalar}\n")
vector_1d = torch.empty(3)
print(f"One dimensional uninitiallised tensor: \n{vector_1d}\n")
vector_2d = torch.empty(2, 3)
print(f"Two dimensional uninitiallised tensor: \n{vector_2d}\n")
# torch.rand(size): random numbers [0, 1]
rand_tensor = torch.rand(5, 3)
print(f"Random tensor: \n{rand_tensor}\n")
# torch.zeros(size), fill with 0
# torch.ones(size), fill with 1
zero_tensor = torch.zeros(5, 3)
print(f"Zero tensor: \n{zero_tensor}\n")
# check tensor size
print(f"Size of tensor: \n{zero_tensor.size()}\n")
# check data type
print(f"Type of tensor: \n{zero_tensor.dtype}\n")
# specify types, float32 default
float_tensor = torch.zeros(5, 3, dtype=torch.float16)
print(f"Float tensor: \n{float_tensor}\n")
print(f"Float tensor type: \n{float_tensor.dtype}\n")
# construct from data
constructed_tensor = torch.tensor([1, 2])
print(f"Constructed tensor size: \n{constructed_tensor.size()}\n")
# tensor with required gradient
grad_tensor = torch.tensor([5.5, 3], requires_grad=True)
print(f"Tensor with gradient: \n{grad_tensor}\n")

# operations on tensors
y = torch.rand(2, 2)
x = torch.rand(2, 2)
# elementwise addition
z = x + y
# z = torch.add(x, y)
# in place addition, everything with a trailing underscore is
# an inplace operation
# y.add_(x)
# subtraction
z = x - y
# z = torch.sub(x, y)
# multiplication
z = x * y
# z = torch.mul(x, y)
# division
z = x / y
# z = torch.div(x, y)

# slicing tensors
x = torch.rand(5, 3)
print(f"Tensor: \n{x}\n")
print(f"Sliced tensor: \n{x[1, :]}\n")
print(f"Tensor value: \n{x[1, 1].item()}\n")

# reshaping tensors
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(f"Initial shape: {x.size()}, Flatten: {y.size()}, Reshape: {z.size()}")

# pytorch tensors and numpy conversions
a = torch.ones(5)
print(f"Tensor: \n{a}")
print(f"Type: \n{type(a)}\n")
b = a.numpy()
print(f"Numpy: \n{b}")
print(f"Type: \n{type(b)}\n")
c = torch.from_numpy(b)
print(f"Tensor: \n{c}")
print(f"Type: \n{type(c)}\n")
