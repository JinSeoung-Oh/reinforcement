# regular fitted Q-iteration

# weight distribution
weights = sa_visitations
network = FCNetwork(env, layers=[20,20])

q_values = fitted_q_iteration(env, network, num_iters=50, discount=0.95, weights=weights, render=True, sampled=False)
v_values = np.max(q_values, axis=1)
plot_s_values(env, v_values, title='Values')

# plot Q-functions, overestimation error

print('Total Error:', np.sum(np.abs(q_values - optimal_qvalues)))
total_overestimation = np.sum((q_values - optimal_qvalues) * weights)
print('Total Weighted Overestimation under the training distribution: ', total_overestimation)

policy = compute_policy_deterministic(q_values, eps_greedy=0)
policy_sa_visitations = compute_visitation(env, policy)
weights_policy = policy_sa_visitations / np.sum(policy_sa_visitations)
total_policy_overestimation = np.sum((q_values-optimal_qvalues) * weights_policy)
print ('Total Overestimation under the learned policy: ', total_policy_overestimation)

plot_sa_values(env, (q_values - optimal_qvalues), title='Q-function Error (Q - Q*)')

policy = compute_policy_deterministic(q_values, eps_greedy=0.1)
policy_sa_visitations = compute_visitation(env, policy)
plot_sa_values(env, policy_sa_visitations, title='Q-hat Visitation')
