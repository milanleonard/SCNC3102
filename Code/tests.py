#%%
"""
This is my attempt at trying to see if we can sort of tackle the QRAM problem
via optimization. I don't know how practical/scalable this approach might be, 
but understanding the optimization routine here might be interesting, plus can
determine exactly what we need to do. 
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim
def vectorkronecker(t1, t2):
    return torch.ger(t1,t2).flatten()

def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))


def cnot_chain_mat(n):
    out = torch.eye(2**n) # output mat
    for i in range(n-1): # This loop goes through each layer in each
        curr = CNOT if i == 0 else torch.eye(2)
        for j in range(n-2): # This loop for each gate in fixed layer
            if i == (j+1):
                curr = kronecker(curr,CNOT)
            else:
                curr = kronecker(curr,torch.eye(2))
        out = out@curr
    return out
CNOT = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

# %%
n = 4
cnot_chain = cnot_chain_mat(n)
assert n%2 == 0



# %%
def getkronecker(x, CNOT_CHAIN=False): # x must be a vector
    out = x[0:2]
    for i in range(1, len(x)//2): # guaranteed currently by global assertion
        out = vectorkronecker(x[i:i+2], out)
    if CNOT_CHAIN:
        out = cnot_chain@out
    return out

def subtract_target(x, target):
    return torch.abs(x-target).mean()

# %%
# INIT PARAMS


"""
EXPERIMENT 1: Tensor product linear subspace of 2^n-1 dim entangled
space
"""
RUNS = 500
runs_loss = []
for run in range(RUNS):
    if run % 10 == 0:
        print(f"Run {run} of {RUNS}")
    target = torch.abs(torch.randn(2**n))
    target = target / torch.norm(target)
    x = Variable(torch.ones(2*n), requires_grad=True)
    optim = torch.optim.Adam([x], lr=0.005)
    for i in range(1000):
        optim.zero_grad()
        out = subtract_target(getkronecker(x), target)
        if i % 500 == 0: print(out)
        out.backward()
        optim.step()
    runs_loss.append(out)
#%%
print("RU")
# %%
"""
Testing that my code could actually even work if I had 
2^n free params 
"""
xprime = Variable(torch.ones(2**n), requires_grad=True)
optim = torch.optim.Adam([xprime], lr=0.0001)
for i in range(10000):
    optim.zero_grad()
    out = torch.mean(torch.abs(xprime-target))
    if (i % 5) == 0: print(out)
    out.backward()
    optim.step()
# %%
"""
(theta_1,theta_2) x (theta_3,theta_4)...(..,theta_2n)
Find thetas just that
it is closest to
(phi1,phi2,...,phi2^n) in computational basis

Construct a general vector, arbitrary unitary on every qubit, then do the cnot chain to enter into maximally entangled state. 

Pick a class of data, relationships between elements in the data. 
"""