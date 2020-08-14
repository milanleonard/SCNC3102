#%%
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim
def vectorkronecker(t1, t2):
    return torch.ger(t1,t2).flatten()
# %%
n = 24
assert n%2 == 0
target = torch.abs(torch.randn(2**n))
target = target / torch.norm(target)
x = Variable(torch.ones(2*n), requires_grad=True)

# %%
class KronckerLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(x): # x must be a vector
        out = x[0:1]
        for i in range(1, x/2): # guaranteed currently by global assertion
            out = kronecker(x[i:i+1], out)


# %%
def getkronecker(x): # x must be a vector
    out = x[0:2]
    for i in range(1, len(x)//2): # guaranteed currently by global assertion
        out = vectorkronecker(x[i:i+2], out)
    return out

def subtract_target(x, target=target):
    return torch.abs(x-target).mean()
# %%
optim = torch.optim.Adam([x], lr=0.1)
for i in range(20):
    optim.zero_grad()
    out = subtract_target(getkronecker(x)) *
    if (i % 5) == 0: print(out)
    out.backward()
    optim.step()
    
# %%
