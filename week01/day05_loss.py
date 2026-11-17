import torch
import torch.nn as nn
import copy

x = torch.linspace(-2,2,21)
x = x.reshape(-1,1)
y = x**2
torch.manual_seed(42)
model_small=nn.Sequential(
    nn.Linear(1,16),
    nn.ReLU(),
    nn.Linear(16,1)
)
model_mid=copy.deepcopy(model_small)
model_big=copy.deepcopy(model_small)
sgd_optimizer_small = torch.optim.SGD(model_small.parameters(),lr=0.0001)
sgd_optimizer_mid = torch.optim.SGD(model_mid.parameters(),lr=0.01)
sgd_optimizer_big = torch.optim.SGD(model_big.parameters(),lr=0.1)
# adam_optimizer = torch.optim.Adam(model_adam.parameters(),lr=0.01)
# print("small_lr:")
for epoch in range(500):
    sgd_optimizer_small.zero_grad()
    y_pred_sgd = model_small(x)
    criterion = nn.MSELoss()
    loss_sgd = criterion(y_pred_sgd,y)
    loss_sgd.backward()
    sgd_optimizer_small.step()
    # if epoch==0 or epoch==100 or epoch==300:
    #     print(f"当前{epoch}轮：SGD_MSE:{loss_sgd}")
# print("mid_lr:")
for epoch in range(500):
    sgd_optimizer_mid.zero_grad()
    y_pred_sgd = model_mid(x)
    criterion = nn.MSELoss()
    loss_sgd = criterion(y_pred_sgd,y)
    loss_sgd.backward()
    sgd_optimizer_mid.step()
    # if epoch==0 or epoch==100 or epoch==300:
    #     print(f"当前{epoch}轮：SGD_MSE:{loss_sgd}")
# print("big_lr:")
for epoch in range(500):
    sgd_optimizer_big.zero_grad()
    y_pred_sgd = model_big(x)
    criterion = nn.MSELoss()
    loss_sgd = criterion(y_pred_sgd,y)
    loss_sgd.backward()
    sgd_optimizer_big.step()
    # if epoch==0 or epoch==100 or epoch==300:
    #     print(f"当前{epoch}轮：SGD_MSE:{loss_sgd}")

model_small.eval()
model_mid.eval()
model_big.eval()

with torch.no_grad():
    criterion=nn.MSELoss()
    y_small = model_small(x)
    y_mid = model_mid(x)
    y_big = model_big(x)
    mse_small=criterion(y_small,y)
    mse_mid=criterion(y_mid,y)
    mse_big=criterion(y_big,y)
    print(f"lr=0.0001 最终MSE：{mse_small}")
    print(f"lr=0.01 最终MSE：{mse_mid}")
    print(f"lr=0.1 最终MSE：{mse_big}")