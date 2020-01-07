import gym
import Agent
import torch as th

class Environment:
    def __init__(self, problem : str):
        self.problem = problem
        self.env = gym.make(problem)

    def run(self, agent : Agent.Agent):
        s = th.tensor(self.env.reset(), dtype=th.float)
        R = 0 

        while True:            
            self.env.render()

            a = agent.act(s)

            s_, r, done, info = self.env.step(a)
            s_ = th.tensor(s_, dtype=th.float)
            
            if done: # terminal state
                s_ = None

            agent.observe( (s, a, r, s_) )
            agent.replay()            

            s = s_
            R += r

            if done:
                break

        print("Total reward:", R)