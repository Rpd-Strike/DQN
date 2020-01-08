import Environment
import Agent
import matplotlib.pyplot as plt

def main():
    PROBLEM = 'CartPole-v1'
    env = Environment.Environment(PROBLEM)

    stateCnt = env.env.observation_space.shape[0]
    actionCnt = env.env.action_space.n


    agent = Agent.Agent(stateCnt, actionCnt)

    lst = []

    while True:
        for _ in range(25):
            lst.append(env.run(agent))
        plt.plot(lst)
        plt.show()

if __name__ == "__main__":
    main()