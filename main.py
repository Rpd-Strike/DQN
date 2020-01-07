import Environment
import Agent

def main():
    PROBLEM = 'CartPole-v1'
    env = Environment.Environment(PROBLEM)

    stateCnt = env.env.observation_space.shape[0]
    actionCnt = env.env.action_space.n


    agent = Agent.Agent(stateCnt, actionCnt)

    
    while True:
        env.run(agent)

if __name__ == "__main__":
    main()