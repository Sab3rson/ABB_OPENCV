import torch
print(torch.__version__)
print(torch.cuda.is_available())       # should be True
print(torch.cuda.get_device_name(0))   # should list your GTX 1660
