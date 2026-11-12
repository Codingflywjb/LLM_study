import torch

print("PyTorch版本:", torch.__version__)
print("CUDA版本:", torch.version.cuda)
print("CUDA可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU型号:", torch.cuda.get_device_name(0))

    a = torch.randn(100, 100, device="cuda")
    b = torch.randn(100, 100, device="cuda")

    c = a @ b

    print("矩阵计算成功:", c.shape)
    print("计算设备:", c.device)