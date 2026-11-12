import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1,1)

    def forward(self,x):
        return self.layer(x)

x=torch.tensor(
    [1,2,3,4,5],
    dtype=torch.float32
)

y=torch.tensor(
    [3,5,7,9,11],
    dtype=torch.float32
)

x=x.reshape(-1,1)
y=y.reshape(-1,1)

model = LinearRegression()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)
for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = torch.mean((y_pred-y)**2)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"当前{epoch}轮，loss:{loss}")

for name,param in model.named_parameters():
    print(name,param.detach())

model.train()
print(model.training)
model.eval()
print(model.training)

x_test = torch.tensor(
    [[6.0], [7.0], [8.0]]
)

with torch.no_grad():
    y_test=model(x_test)
    print(y_test)
    print(y_test.shape)
    y_eval = torch.tensor([13.0,15.0,17.0])
    y_eval = y_eval.reshape(-1,1)
    loss=torch.mean((y_test-y_eval)**2)
    print(loss)
    print(y_test.requires_grad)

