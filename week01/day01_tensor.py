import torch

#任务一：了解张量
print("任务一：")
scores = torch.tensor(
    [
        [88,92],
        [76,81],
        [95,90]
    ],dtype=torch.float32
)

print(scores.shape)
print(scores.ndim)
print(scores.dtype)
print(scores.device)

print("学生A的两科成绩分别为：",scores[0])  ##scores[0]即可表示学生A的全部成绩

print("各学生数学成绩如下",scores[:,0])  ##利用tensor切片代替循环

print("任务二：")
##任务二：学习张量基本计算
a = torch.tensor([
    [1,2],
    [3,4]
],device="cpu")
b=torch.tensor([
    [5,6],
    [7,8]
],device="cpu")

print(a+b)
print(a*b)
print(a@b)

print("任务三：")
##任务三：学习自动求导
x=torch.tensor(3.0,requires_grad=True)
y=x*x+2*x+1
y.backward()
print(y)
print(x.grad)

print("任务四：")
##计算图与链式法则
x=torch.tensor(2.0,requires_grad=True)
a=3*x+1
y=a*a
y.backward()
print(a)
print(y)
print(x.grad)
print(a.grad_fn)
print(y.grad_fn)

print("任务五：")
#实现梯度下降
w=torch.tensor(0.0,requires_grad=True)
lr=0.1
for epoch in range(20):
    loss=(w-3)*(w-3)
    loss.backward()
    with torch.no_grad(): ##参数更新不是模型前向计算的一部分，我们不希望 PyTorch 再为参数更新建立新的计算图。
        w-=lr*w.grad
    w.grad.zero_() ##PyTorch 默认会累积梯度,假设第一次梯度为 6，第二次梯度为 4，如果没有清零，那么第二次 w.grad 可能累积为 10，而不是单独的 4。
    print(epoch)
    print(w)
    print(loss)
