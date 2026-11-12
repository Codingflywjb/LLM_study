import torch
import torch.nn as nn

# 任务一
# x = torch.tensor([-5.0, -2.0, 0.0, 3.0, 7.0])
#
# relu = nn.ReLU()
#
# y=relu(x)
#
# print(f"处理前：{x},处理后：{y}")
# print(f"处理前shape：{x.shape},处理后：{y.shape}")

##进阶挑战
# relu=torch.where(x>0,x,0)

# 任务二：构建简单的隐藏层
# x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]],dtype=torch.float32)
#
# model = nn.Sequential(
#     nn.Linear(1,8),
#     nn.ReLU(),
#     nn.Linear(8,1)
# )
#
# y_pred = model(x)
#
# print(model)
# print(y_pred)
# print(y_pred.shape)
#
# for name,param in model.named_parameters():
#     print(name,param.detach())

# 任务三：构建非线性训练数据
x = torch.linspace(-2,2,21)
x = x.reshape(-1,1)
y = x**2

linear_model = nn.Linear(1,1)
linear_optimizer = torch.optim.SGD(
    linear_model.parameters(),
    lr=0.01
)

for epoch in range(2000):
    linear_optimizer.zero_grad()
    linear_y_pred = linear_model(x)
    linear_loss = torch.mean((linear_y_pred-y)**2)
    linear_loss.backward()
    linear_optimizer.step()


model = nn.Sequential(
    nn.Linear(1,16),
    nn.ReLU(),
    nn.Linear(16,1)
)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)
# print(f"{x.shape},{y.shape}")
for epoch in range(2000):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = torch.mean((y_pred-y)**2)
    loss.backward()
    optimizer.step()
    # if epoch % 200 == 0:
    #     print(f"当前{epoch}轮：loss={loss}")

linear_model.eval()
model.eval()
with torch.no_grad():
    x_test = torch.tensor([
        [-2.0],
        [0.0],
        [2.0]
    ])
    y_linear=linear_model(x)
    y_mlp=model(x)
    linear_mse = torch.mean((y_linear-y)**2)
    mlp_mse = torch.mean((y_mlp-y)**2)
    print(f"Linear 训练集 MSE:{linear_mse}")
    print(f"MLP 训练集 MSE:{mlp_mse}")
    print(f"Linear 对 [-2,0,2] 的预测:{linear_model(x_test)}")
    print(f"MLP 对 [-2,0,2] 的预测:{model(x_test)}")
# print(y_pred.shape)
# print(loss)
