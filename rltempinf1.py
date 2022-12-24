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
from jtop import jtop
import RPi.GPIO as GPIO
import time
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import policy_saver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

output_dir = "model_output/cabintemp/"
#https://www.tensorflow.org/agents/tutorials/10_checkpointer_policysaver_tutorial

def main():
    converter = tf.lite.TFLiteConverter.from_saved_model(policy_dir, signature_keys=["action"])
    tflite_policy = converter.convert()
    with open(os.path.join(output_dir, 'policy.tflite'), 'wb') as f:
      f.write(tflite_policy)

import numpy as np
interpreter = tf.lite.Interpreter(os.path.join(output_dir, 'policy.tflite'))

policy_runner = interpreter.get_signature_runner()
print(policy_runner._inputs)


