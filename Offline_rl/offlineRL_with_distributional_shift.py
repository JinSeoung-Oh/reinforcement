import numpy as np
from Environment import spec_from_string
from Environment import GridEnv
from Qlearning import q_iteration, conservative_q_iteration
from util import compute_policy_deterministic, compute_visitation
from plotting import plot_sa_values, plot_s_values
from NN import FCNetwork

#input parameter

weighting_only = True #@param {type: "boolean"}
dataset_composition = 'random+optimal' #@param ["optimal", "random", "random+optimal", "mixed"]
dataset_size =  100#@param {type: "integer"}
env_type = 'smooth' #@param ["smooth", "random", "onehot"]

#Define env and comput optimal q-value
maze = spec_from_string("SOOOOOO#\\"+
                        "O##O###O\\"+
                        "OO#OO##O\\"+
                        "O#RO#OO#\\"
                       )

env = GridEnv(maze, observation_type=env_type, dim_obs=8)
optimal_qvalues = q_iteration(env, num_itrs=100, discount=0.95, render=False)

plot_sa_values(env, optimal_qvalues, title='Q*-values')

policy = compute_policy_deterministic(optimal_qvalues, eps_greedy=0.1)
sa_visitations = compute_visitation(env, policy)
plot_sa_values(env, sa_visitations, title='Optimal policy state-action visitation')

# comput weights
if dataset_composition == 'optimal':
  """Distribution of the optimal policy (+ some noise)"""
  weights = sa_visitations
  weights = weights/ np.sum(weights)
elif dataset_composition == 'random':
  """A random disribution over states and actions""" 
  weights = np.random.uniform(size=env.num_states * env.num_actions)
  weights = np.reshape(weights, (env.num_states, env.num_actions))
  weights = weights/ np.sum(weights)
elif dataset_composition == 'random+optimal':
  """Mixture of random and optimal policies"""
  weights = sa_visitations / np.sum(sa_visitations)
  weights_rand = np.random.uniform(size=env.num_states * env.num_actions)
  weights_rand = np.reshape(weights_rand, (env.num_states, env.num_actions)) / np.sum(weights_rand)
  weights = (weights_rand + weights)/2.0
elif dataset_composition == 'mixed':
  """Mixture of policies corresponding to random Q-values"""
  num_policies_mix = 4
  weights = np.zeros_like(sa_visitations)
  for idx in range(num_policies_mix):
    rand_q_vals_idx = np.random.uniform(low=0.0, high=10.0, size=(env.num_states, env.num_actions))
    policy_idx = compute_policy_deterministic(rand_q_vals_idx, eps_greedy=0.1)
    sa_visitations_idx = compute_visitation(env, policy_idx)
    weights = weights + sa_visitations_idx
  weights = weights / np.sum(weights)

# Generate dataset
if not weighting_only:
  weights_flatten = np.reshape(weights, -1)
  weights_flatten = weights_flatten/ np.sum(weights_flatten)
  dataset = np.random.choice(
      np.arange(env.num_states * env.num_actions),
      size=dataset_size, replace=True, p=weights_flatten
  )
  training_sa_pairs = [(int(val//env.num_actions), val % env.num_actions) for val in dataset]

  # Now sample (s', r) values for training as well
  training_dataset = []
  training_data_dist = np.zeros((env.num_states, env.num_actions))
  for idx in range(len(training_sa_pairs)):
    s, a = training_sa_pairs[idx]
    prob_s_prime = env._transition_matrix[s, a]
    s_prime = np.random.choice(np.arange(env.num_states), p=prob_s_prime)
    r = env.reward(s, a, s_prime)
    training_dataset.append((s, a, r, s_prime))
    training_data_dist[s, a] += 1.0
else:
  # Using only weighting style dataset
  training_dataset = None
  training_data_dist = None

#vis dataset or weights
if not weighting_only:
  plot_sa_values(env, training_data_dist, title='Dataset composition')
else:
  plot_sa_values(env, weights, title='Weighting Distribution')


## regular fitted Q_iteration with finite data

network = FCNetwork(env, layers=[20,20])
q_values = fitted_q_iteration(env, network, num_itrs=100, discount=0.5, weights=weights, render=True, sampled=not(weighting_only),training_dataset=training_dataset)

v_values = np.max(q_values, axis=1)
polt_s_values(env, v_values, title='Values')


print('Total Error:', np.sum(np.abs(q_values - optimal_qvalues)))
total_overestimation = np.sum((q_values - optimal_qvalues)*weights)
print('Total Weighted Overestimation under the training distribution: ', total_overestimation)

policy = compute_policy_deterministic(q_values, eps_greedy=0.1)
policy_sa_visitations = compute_visitation(env, policy)
weights_policy = policy_sa_visitations/np.sum((q_values - optiman_qvaules)*weights_policy)
print ('Total Overestimation under the learned policy: ', total_policy_overestimation)

total_overestimation_unweighted = np.mean((q_values - optimal_qvalues))
print ('Total Overestimation: ', total_overestimation_unweighted)

plot_sa_values(env, (q_values - optimal_qvalues), title='Q-function Error (Q - Q*)')

policy = compute_policy_deterministic(q_values, eps_greedy=0.1)
policy_sa_visitations = compute_visitation(env, policy)
plot_sa_values(env, policy_sa_visitations, title='Q-hat Visitation')


## Run conservative Q-iteration (or CQL) with finite data

network = FCNetwork(env, layers=[20,20])
cql_alpha_val = 0.1
print (weighting_only)

q_values = conservative_q_iteration(env, network,
                                    num_itrs=100, discount=0.95, cql_alpha=cql_alpha_val, 
                                    weights=weights, render=True,
                                    sampled=not(weighting_only),
                                    training_dataset=training_dataset)

v_values = np.max(q_values, axis=1)
plot_s_values(env, v_values, title='Values')

print('Total Error:', np.sum(np.abs(q_values - optimal_qvalues)))

total_overestimation = np.sum((q_values - optimal_qvalues) * weights)
print('Total Weighted Overestimation under the training distribution: ', total_overestimation)


policy = compute_policy_deterministic(q_values, eps_greedy=0.1)
policy_sa_visitations = compute_visitation(env, policy)
weights_policy = policy_sa_visitations / np.sum(policy_sa_visitations)
total_policy_overestimation = np.sum((q_values - optimal_qvalues) * weights_policy)
print ('Total Overestimation under the learned policy: ', total_policy_overestimation)

total_overestimation_unweighted = np.mean((q_values - optimal_qvalues))
print ('Total Overestimation: ', total_overestimation_unweighted)

plot_sa_values(env, (q_values - optimal_qvalues), title='Q-function Error (Q - Q*)')


policy = compute_policy_deterministic(q_values, eps_greedy=0.1)
policy_sa_visitations = compute_visitation(env, policy)
plot_sa_values(env, policy_sa_visitations, title='Q-hat_CQL Visitation')
