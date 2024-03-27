import numpy as np
import torch 

# def softmax(z):
#     t = np.exp(z)
#     a = np.exp(z) / np.expand_dims(np.sum(t, axis = -1), -1)
#     return a

# Query = np.array([
#     [1, 0, 2],
#     [1, 0, 2],
#     [1, 0, 2]
# ])

# Key = np.array([
#     [2, 0, 1],
#     [0, 0, 3],
#     [2, 1, 2]

# ])

# scores = Query @ Key.T
# print(scores)
# print(softmax(0.5))

x  = torch.randn(1, 256, 40, 40 )
m = torch.nn.AvgPool2d(kernel_size=1)
y = m(x)
y = torch.squeeze(x, 1)
# print(x.size())
# reshape = torch.reshape(y, (1, 256, 1, 40))
# print(reshape.shape)
# print(m(x).shape[2])
x = torch.zeros(1, 512, 40, 40)
x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
# x_h = torch.mean(x, dim=3, keepdim=True)
x_w = torch.mean(x, dim=2, keepdim=True)
print(x_h.shape)
print(x_w.shape)
print(torch.cat((x_h, x_w), 3).shape)
# print(y.shape)