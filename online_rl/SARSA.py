import numpy as np
import gym

env = gum.make('FrozenLake-v0')

epsilon = 0.9
total_episodes = 1000
max_steps = 100
alpha = 0.85
gamma = 0.95

Q = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state):
  action = 0
  if np.random.uniform(0,1) < epslion:
    action = env.action_space.sample()
  else:
    action = np.argmax(Q[state,:])
  return action

def update(state, state2, reward, action, action2):
  predict = Q[state, action]
  target = reward + gamma * Q[state2, action2]
  Q[state, action] = Q[state, action] + alpha * (target - predict)


reward = 0
for episode in ragne(total_episodes):
  t = 0
  state1 = env.reset()
  action1 = choose_action(state1)

  while t < mac_steps:
    env.render()
    state2, reward, done, info = env.step(action1)
    action2 = choose_action(state2)
    t += 1
    reward +=1
    if done:
      break


# evaluating
print("performance: ", reward/total_episodes)
print(Q)
