import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make('CartPole-v1')

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from Ipython import display

plt.ion()

device = torch.device("cuda" if torch.cuda.is_availble() else "cpu")
