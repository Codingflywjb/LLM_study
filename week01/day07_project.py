import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader

torch.manual_seed(42)
x = torch.linspace(-2,2,41).reshape(-1,1)
y = x ** 2

indices = torch.randperm(41)

train_idx = indices[:32]
val_idx = indices[32:]

x_train = x[train_idx]
y_train = y[train_idx]
x_val = x[val_idx]
y_val = y[val_idx]

dataset_train = TensorDataset(x_train,y_train)
dataset_val = TensorDataset(x_val,y_val)

loader_train = DataLoader(
    dataset_train,
    batch_size=8,
    shuffle=True
)
# print(len(loader_train))

loader_val = DataLoader(
    dataset_val,
    batch_size=8,
    shuffle=False
)
# print(len(loader_val))

model = nn.Sequential(
    nn.Linear(1,16),
    nn.ReLU(),
    nn.Linear(16,1)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
# print(model)
# params = []
# for param in model.parameters():
#     params.append(param.numel())
# print(sum(params))

# print(f"x_train: {x_train.shape}")
# print(f"y_train: {y_train.shape}")
# print(f"x_val: {x_val.shape}")
# print(f"y_val: {y_val.shape}")

# print(f"训练数据匹配：{torch.allclose(y_train,x_train**2)}")
# print(f"验证数据匹配：{torch.allclose(y_val,x_val**2)}")

best_val_mse = float("inf")
best_epoch = 0

for epoch in range(200):
    model.train()
    for x_batch,y_batch in loader_train:
        optimizer.zero_grad()
        y_batch_pred = model(x_batch)
        loss = criterion(y_batch_pred,y_batch)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        mse_train = criterion(model(x_train),y_train)
        total_loss = 0.0
        total_val = 0.0
        for x_batch_val,y_batch_val in loader_val:
            batch_loss = criterion(model(x_batch_val),y_batch_val)
            total_loss += batch_loss.item() * len(x_batch_val)
            total_val += len(x_batch_val)
        mse_val = total_loss / total_val
        if mse_val < best_val_mse:
            best_val_mse = mse_val
            best_epoch = epoch+1
            torch.save(model.state_dict(),"day07_best_model.pth")
    if epoch % 50 == 0:
        print(f"Epoch {epoch}：Train MSE = {mse_train}，Val MSE：{mse_val}")
# print(best_epoch)
# print(best_val_mse)
# print(mse_val)

best_model = nn.Sequential(
    nn.Linear(1,16),
    nn.ReLU(),
    nn.Linear(16,1)
)

state = torch.load("day07_best_model.pth",weights_only=True)

best_model.load_state_dict(state)

best_model.eval()
with torch.no_grad():
    x_test = torch.tensor([
        [-1.5],
        [0.5],
        [1.5]
    ])
    y_val_pred = best_model(x_val)
    mse = criterion(y_val_pred,y_val)
    print(f"Val MSE：{mse}")
    print(f"对比结果：{torch.allclose(mse,torch.tensor(best_val_mse))}")
    print(f"[-1.5,0.5,1.5]预测值：{best_model(x_test)}")


