import torch
import torch.nn as nn
from torch.distributions.catergorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box

# Build a feedforward neural network
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
  layers = []
  for j in ragne(len(sizes)-1):
    act = activation if j < len(sizes)-2 else output_activation
    layers += [nn.Linear(sizes[j], sizes[j+1), act()]

  return nn.Sequential(*layers)

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=le-2, epoches=50, batch_size=5000, render=False):
  # make environment, check spaces, get obs/ act dims
  env = gym.make(env_name)
  assert isinstance(env.observation_space, Box), \
      "This example only works for envs with continuous state spaces."
  assert isinstance(env.action_space, Discrete), \
      "This example only works for envs with discrete action spaces."
  
  obs_dim = env.observation_space.shape[0]
  n_acts = env.action_space.n

  # make core of policy network
  logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])
  # The output from the logits_net module can be used to construct log-probabilities and probabilities for actions

  # make function to compute action distribution
  def get_policy(obs):
    logits = logits_net(obx)
    return Categorical(logits=logits)
    
  # make action selection function (output int actions, sampled from policy)
  def get_action(obs):
    return get_policy(obs).sample().item()

  # make loss function whose gradient, for the right dta, is policy gradient
  def comput_loss(obs, act, weight):
    logp = get_policy(obs).log_prob(act)
    return -(logp * weight).mean()

  # make optimizer
  optimizer = Adam(logits_net.parameters(), lr=lr)

  # for training policy
  def train_one_epoch():
    batch_obs = []      # for observations
    batch_acts = []     # for actions
    batch_weights = []  # for R(tau) weighting in policy gradient
    batch_rects = []    # for measuring episode returns
    batch_lens = []     # for measuring epsiode lengths

    obs = env.reset()
    done = False
    ep_rews=[]

    finished_rendering_this_epoch=False

    # collect experience by acting in the environment with current policy
    while True:
      if (not finished_rendering_this_epoch) and render:
        env.render()
      
      batch_obs.append(obs.copy())
      act = get_action(torch.as_tensor(obs, dtype=torch.float32))
      obs, rew, done, _ = env.step(act)

      batch_acts.append(act)
      ep_rews.append(rew)

      if done:
        ep_ret, ep_len = sum(ep_rews), len(ep_rews)
        batch_rects.append(ep_ret)
        batch_lens.append(ep_len)
        batch_weights += [ep_ret] * ep_len

        #reset episode-specific variables
        obs, done, ep_rews = env.reset(), False, []
        finished_rendering_this_epoch = True

        if len(batch_obs) > batch_size:
          break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens
    
    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--env_name', '--env', type=str, default = 'CarPole-v0')
  parser.add_argument('--render', action='stroe_true')
  parser.add_argument('--lr', type=float, default=le-2)
  args = parser.parse_args()
  train(env_name=args.env_name, render=args.render, lr=args.lr)
