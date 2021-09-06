import torch

a = torch.tensor([[4,3],[2,1],[5,7], [2,3]])
b = torch.tensor([0,0,1,0])

_, preds = torch.max(a, dim=1)
print(preds)

sum = torch.sum(preds == b).item()
print(sum)

t = torch.tensor(torch.sum(preds == b).item() / len(preds))

print(t)