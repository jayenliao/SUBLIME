import torch 

i = [[1, 1]]
v =  [3, 4]
s = torch.sparse_coo_tensor(i, v, (3,))
print(s)

print(s.coalesce())