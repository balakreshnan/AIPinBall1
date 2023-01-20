import numpy as np
import gym
from gym import spaces
import logging
import numpy as np
import random
import numpy as np
from collections import deque
from keras.models import Sequential 
from keras.layers import Dense
from keras.optimizers import Adam
import os
#from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
#from jtop import jtop
#import RPi.GPIO as GPIO
import time


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
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
obs = env.reset()
env.render()

score = random.randrange(50, 100000, 3)
previsouscore= 0
ballzonelist = ['left', 'right']
ballzone = random.choice(ballzonelist)

batch_size = 32
n_episodes = 100
output_dir = "model_output/cabintemp/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()
 
    def _build_model(self):
        model = Sequential() 
        model.add(Dense(32, activation="relu",
                        input_dim=self.state_size))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse",
                     optimizer=Adam(lr=self.learning_rate))
        return model
 
    def remember(self, state, action, reward, next_state, done): 
        self.memory.append((state, action,
                            reward, next_state, done))

    def train(self, batch_size):
         minibatch = random.sample(self.memory, batch_size)
         for state, action, reward, next_state, done in minibatch:
            target = reward # if done 
            if not done:
                target = (reward +
                          self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) 
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size) 
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def save(self, name): 
        self.model.save_weights(name)

def main():
    agent = DQNAgent(state_size, action_size)

    def getcurrentstatus():
        score = random.randrange(50, 100000, 3)
        previsouscore= 0
        ballzone = 'left'

        return score, ballzone, previsouscore

    state = 0
    previsouscore= 0
    ballzonelist = ['left', 'right']


    for e in range(n_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        done = False 
        time = 0
        while not done:
            #env.render()
            action = agent.act(state)
            score = random.randrange(50, 100000, 3)
            
            #input either left or right
            #ballzone = random.choice(ballzonelist)
            ballzone = 'Left' #pass value from image model
            # print(env)
            # obs, reward, done, info = env.step(cabintemp , outsidetemp, surfacetemp,outlettemp)
            next_state, reward, done, _ = env.step(score , ballzone, previsouscore)
            reward = reward if not done else 0
            next_state = np.reshape(next_state, [1, state_size]) 
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            done = True
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}, reward: {}, state: {}"
                    .format(e, n_episodes-1, time, agent.epsilon, reward, next_state[0]))
            time += 1
            previsouscore = score
            if next_state[0] >= 1:
                curr_value = 'left flap'
                #GPIO.output(output_pin, curr_value)
            else:
                curr_value = 'Right flap'
                #GPIO.output(output_pin, curr_value)
        if len(agent.memory) > batch_size:
            agent.train(batch_size) 
        if e % 50 == 0:
            agent.save(output_dir + "weights_"
                    + "{:04d}".format(e) + ".hdf5")

if __name__ == "__main__":
   main()