import torch as th
import torch.nn as nn
import numpy as np

class Brain(nn.Module):
    def __init__(self, StateCnt, ActionCnt):
        super().__init__()
        self.StateCnt = StateCnt
        self.ActionCnt = ActionCnt

        mij_s = 64
        self.model = nn.Sequential(
            nn.Linear(StateCnt, mij_s),
            nn.Tanh(),
            nn.Linear(mij_s, ActionCnt)
        )
        self.optimizer = th.optim.Adam(self.parameters(), lr = 5e-3)

    def predict(self, s : th.Tensor):
        return self.model(s)
    
    def predictOne(self, s : th.Tensor):
        return self.model(th.stack([s]))[0]

    def train(self, x : th.Tensor, y : th.Tensor):
        self.optimizer.zero_grad()
        loss = th.norm(self.model(x) - y)
        loss.backward()
        self.optimizer.step()
