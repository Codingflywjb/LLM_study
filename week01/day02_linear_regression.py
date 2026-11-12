import torch


print("任务一：")
##训练数据与损失计算
print("任务二：")
##SGD训练
x=torch.tensor(
    [1,2,3,4,5],
    dtype=torch.float32
)

y=torch.tensor(
    [3,5,7,9,11],
    dtype=torch.float32
)

# w=0.0
# b=0.0
w=torch.tensor(0.0,requires_grad=True)
b=torch.tensor(0.0,requires_grad=True) ##w,b必须是能够参与自动求导的 Tensor。

optimizer = torch.optim.SGD([w,b],lr=0.01)
# SGD 全称为 Stochastic Gradient Descent，随机梯度下降。
# 它可以按照梯度自动调整模型参数.
# 使用 PyTorch 时，只需要告诉优化器：
# - 哪些参数需要训练。
# - 学习率是多少。
for epoch in range(1000):
    ## 训练循环顺序
    # 清零梯度：optimizer.zero_grad()
    # 前向传播：计算y_pred
    # 计算损失：计算loss
    # 反向传播：loss.backward()
    # 更新参数：optimizer.step()

    # w.grad.zero_()
    # b.grad.zero_()  没必要
    optimizer.zero_grad()  ##它会清理优化器所管理参数的梯度
    y_pred = w*x+b
    loss=torch.mean((y_pred-y)**2)  ##MSE均方误差计算
    loss.backward()
    optimizer.step()  ##torch.no_grad() 是多余的,因为 PyTorch 的常规优化器在执行 step() 时已经会避免记录参数更新的计算图。

    if epoch % 100 == 0:
        print(f"当前轮数{epoch}/1000: w:{w.item()}  b:{b.item()}  loss:{loss.item()}")
print(f"训练完成：最终w：{w.item()},b:{b.item()}")

x_new = torch.tensor(6.0,dtype=torch.float32)

x_test = torch.tensor([6.0,7.0,8.0])

with torch.no_grad():
    y_new=w*x_new+b
    y_test_new=w*x_test+b
    print(y_new.item())
    print(y_test_new)
