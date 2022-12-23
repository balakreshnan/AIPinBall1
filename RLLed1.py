import numpy as np
import gym
from gym import spaces
import logging
import numpy as np
import random
import gym
import numpy as np
from collections import deque
#from keras.models import Sequential 
#from keras.layers import Dense
#from keras.optimizers import Adam
import os
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

class PinballSimulatorEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    cabintemp = 65
    outsidetemp = 66
    surfacetemp = 70
    outlettemp =72
    action = 2

    def __init__(self):
        self.logger = logging.getLogger("Logger")
        self.step_logger = logging.getLogger("StepLogger")
        self.__version__ = "0.0.1"
        self.logger.info(f"PinballSimulatorEnv - Version {self.__version__}")
        # Define the action_space
        # Set temp 
        # action 
        # 1 = increase
        # 2 = decrease
        # = = do nothing
        n_actions = 1
        self.action_space = spaces.Discrete(n_actions)

        # Define the observation_space
        # First dimension is CabinTemp (50..78)
        # Second dimension is outside temp in the air (10...110)
        # Third dimension is surface temperature (10..100)
        # Fourth dimension is outlet temperature (50..100)
        # Fifth dimension is climate temperature (30..100)
        low = np.array([0, 400, -100])
        high = np.array([3, 3000, 100])
        self.observation_space = spaces.Box(low=0, high=3,
                                        shape=(1,), dtype=np.float32)
    
    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        # Initialize the agent at the right of the grid
        self.agent_pos = 0
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([self.agent_pos]).astype(np.float32)

    def step(self, score , ballzone, previsouscore):
        action = 0
        if ballzone == 'left':
            action = 1
        elif ballzone == 'right':
            action = 2
        else:
            action = 0
            # raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        # print('cabintemp=', cabintemp, 'outsidetemp=', outsidetemp, 'surfacetemp=', surfacetemp, 'outlettemp=', outlettemp, 'Action=', action)
        # Account for the boundaries of the grid
        # self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

        # Are we at the left of the grid?
        # done = bool(self.agent_pos == 0)
        done = False

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = 1 if score >= previsouscore else 0

        # Optionally we can pass additional info, we are not using that for now
        info = {}
        # print(self.action, reward, done, info)

        return np.array([action]).astype(np.float32), reward, done, info
    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        print("." * self.agent_pos, end="")
        print("x", end="")
        # print("." * (self.grid_size - self.agent_pos))

    def close(self):
        pass
        

env = PinballSimulatorEnv()

aggr_ep = { 'obs': [], 'reward': [], 'step': [], 'score': []}

import random

aggr_ep = { 'obs': [], 'reward': [], 'step': [], 'score': []}

env = PinballSimulatorEnv()

obs = env.reset()
env.render()

score = random.randrange(50, 100000, 3)
previsouscore= 0
ballzonelist = ['left', 'right']
ballzone = random.choice(ballzonelist)



print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())

GO_LEFT = 0
# Hardcoded best agent: always go left!
n_steps = 500
for step in range(n_steps):
  print("Step {}".format(step + 1))
  score = random.randrange(50, 100000, 3)
  ballzone = random.choice(ballzonelist)
  # print(env)
  obs, reward, done, info = env.step(score , ballzone, previsouscore)
  # print('score=', score, 'ballzone=', ballzone, 'previsouscore=', previsouscore)
  print('obs=', obs[0], 'reward=', reward, 'done=', done)
  aggr_ep['obs'].append(obs[0])
  aggr_ep['reward'].append(reward)
  aggr_ep['step'].append(step)
  aggr_ep['score'].append(score)
  env.render()
  previsouscore = score
  # if done:
  #  print("Goal reached!", "reward=", reward)
  # break
  
plt.figure(figsize=(12, 5))
plt.plot(aggr_ep['step'], aggr_ep['obs'], label="action")
plt.plot(aggr_ep['step'], aggr_ep['reward'], label="rewards")
plt.plot(aggr_ep['step'], aggr_ep['score'], label="score")
plt.legend(loc=4)
plt.show()
