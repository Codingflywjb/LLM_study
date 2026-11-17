import torch
from torch.utils.data import TensorDataset,DataLoader
import torch.nn as nn


x = torch.linspace(-2,2,21).reshape(-1,1)
y = x ** 2

dataset = TensorDataset(x,y)
# print(len(dataset))
# print(dataset[0])
# print(dataset[-1])

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)
# for batch_idx,(x_batch,y_batch) in enumerate(loader,start=1):
#     print(f"Batch {batch_idx}")
#     print("x shape:", x_batch.shape)
#     print("y shape:", y_batch.shape)
#     print( "标签匹配：" , torch.allclose( x_batch ** 2 , y_batch ) )

model = nn.Sequential(
    nn.Linear(1,16),
    nn.ReLU(),
    nn.Linear(16,1)
)

global_step = 0
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(),lr=0.01)

for epoch in range(200):
    for x_batch,y_batch in loader:
        optimizer.zero_grad()
        y_batch_pred = model(x_batch)
        loss = criterion(y_batch_pred,y_batch)
        loss.backward()
        optimizer.step()
        global_step += 1
    if epoch % 50 == 0:
        with torch.no_grad():
            y_pred = model(x)
            print(f"当前{epoch}轮：MSE为{criterion(y_pred,y).item()},当前global_step：{global_step}")

print(f"最终global_step：{global_step}")
print(len(loader))
model.eval()
with torch.no_grad():
    x_test = torch.tensor([
        [-2.0],
        [0.0],
        [2.0]
    ])
    y_final_pred = model(x)
    print(f"最终MSE为{criterion(y_final_pred,y)}")
    print(f"[-2,0,2]预测为:{model(x_test)}")

state = model.state_dict()
for name,value in state.items():
    print(name,value.shape)
torch.save(state,"day06_mlp.pth")

new_model = nn.Sequential(
    nn.Linear(1,16),
    nn.ReLU(),
    nn.Linear(16,1)
)
state = torch.load(
    "day06_mlp.pth",
    weights_only=True
)
new_model.load_state_dict(state)
new_model.eval()
with torch.no_grad():
    x_test = torch.tensor([
        [-2.0],
        [0.0],
        [2.0]
    ])
    y_pred_initial = model(x_test)
    y_pred_new = new_model(x_test)
    print(f"原模型预测：{y_pred_initial}")
    print(f"新模型预测：{y_pred_new}")
    print(f"是否一致：{torch.allclose(y_pred_new,y_pred_initial)}")