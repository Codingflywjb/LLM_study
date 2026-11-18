import torch
import torch.nn as nn

# x = torch.arange(24).reshape(2,3,4)
#
# # print(x.shape)
# # print(x.ndim)
# # print(x.numel())
# # print(x[0])
# # print(x[0,1].shape)
# # print(x[1,2,:])
# # print(x[1,2,3].shape)
#
# # a = x.reshape(2,4,3)
# # b = x.transpose(1,2)
# # c = x.permute(2,0,1)
# # print(f"{a.shape},{b.shape},{c.shape}")
# # print(f"{a[0]},{b[0]},{c[0]}")
#
# A = torch.tensor([
#     [1., 2., 3.],
#     [4., 5., 6.]
# ])
#
# B = torch.tensor([
#     [1., 2.],
#     [3., 4.],
#     [5., 6.]
# ])
#
# C = torch.matmul(A, B)
#
# print(C)
# print(C.shape)
#
# x_T = x.transpose(1,2)
#
# scores = torch.matmul(x,x_T)
# print(x.shape)
# print(x_T.shape)
# print(scores.shape)
# print(scores[0])

torch.manual_seed(42)
x = torch.arange(24,dtype=torch.float32).reshape(2,3,4)

q_proj = nn.Linear(4,4)
k_proj = nn.Linear(4,4)
v_proj = nn.Linear(4,4)

q = q_proj(x)
k = k_proj(x)
v = v_proj(x)

k_T = k.transpose(1,2)
scores = torch.matmul(q,k_T)
output = torch.matmul(scores,v)

print(x.shape)
print(q.shape)
print(k.shape)
print(v.shape)
print(k_T.shape)
print(scores.shape)
print(output.shape)
print(output[0])


# layer = nn.Linear(4,6)
# y = layer(x)
#
# print(x.shape)
# print(y.shape)
# print(layer.weight.shape)
# print(layer.bias.shape)
# print(layer.weight.numel()+layer.bias.numel())
#
# single_token = x[0,0]
#
# y_pred = layer(single_token)
# print(f"是否一致:{torch.allclose(y_pred,y[0,0])}")


