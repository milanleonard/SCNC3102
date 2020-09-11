#%%
import torch
from torch.nn import functional as F
from torch import nn
from torch import optim
import pandas as pd
import numpy as np
# %%
data = np.genfromtxt('fashion-mnist_test.csv', delimiter=',')
#%%
labels, data = torch.LongTensor(data[1:,0].astype(int)), torch.Tensor(data[1:,1:])
# %%
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(784,5)
        self.fc2 = nn.Linear(5,10)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.relu(self.fc2(x))
# %%
Net = NN()
opt = optim.SGD(Net.parameters(), lr=0.01, momentum=0.9)
loss = nn.CrossEntropyLoss()
# %%
for idx, (x, label) in enumerate(zip(data, labels)):
    opt.zero_grad()
    output = Net(x)
    l = loss(output.unsqueeze(0), label.unsqueeze(0))
    if idx % 10 == 0: print(l)
    l.backward()
    opt.step()
# %%

# %%
