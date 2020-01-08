import numpy as np
import Brain
import Memory
import math
import random
import torch as th

MEMORY_CAPACITY = 1000000
BATCH_SIZE = 64
GAMMA = 0.99
MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain.Brain(stateCnt, actionCnt)
        self.memory = Memory.Memory(MEMORY_CAPACITY)
        
    def act(self, s : th.tensor):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return int(th.argmax(self.brain.predictOne(s)))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)        

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)
        no_state = th.zeros(self.stateCnt)

        states = th.stack([ o[0] for o in batch ])
        states_ = th.stack([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)

        x = th.zeros((batchLen, self.stateCnt))
        y = th.zeros((batchLen, self.actionCnt))
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]
            a = o[1]
            r = o[2]
            s_ = o[3]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * th.max(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)