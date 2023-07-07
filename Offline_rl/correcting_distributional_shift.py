import numpy


network = FCNetwork(env, layers=[20, 20])

cql_alpha_val = 1 # @param {type:"slider", min:0.0, max:10.0, step:0.1}

# Run Q-iteration
q_values = conservative_q_iteration(env, network,
                                    num_itrs=50, discount=0.95, cql_alpha=cql_alpha_val, 
                                    weights=weights, render=True)

# Compute and plot the value function
v_values = np.max(q_values, axis=1)
plot_s_values(env, v_values, title='Values')

#plot q_functions, overestimation error

print('Total Error:', np.sum(np.abs(q_values - optimal_qvalues)))

total_overestimation = np.sum((q_values-optimal_qvalues)*weights)
print('Total Weighted Overestimation under the training distribution: ', total_overestimation)

policy = compute_policy_deterministic(q_values, eps_greedy=0)
policy_sa_visitations = compute_visitation(env, policy)
weights_policy = policy_sa_visitations/np.sum((q_values - optimal_qvalues) * weights_policy)
print ('Total Overestimation under the learned policy: ', total_policy_overestimation)

total_overestimation_unweighted = np.mean((q_values-optimal_qvalues))
print ('Total Overestimation: ', total_overestimation_unweighted)
plot_sa_values(env, (q_values - optimal_qvalues), title='Q-function Error (Q - Q*)')


policy = compute_policy_deterministic(q_values, eps_greedy=0.1)
policy_sa_visitations = compute_visitation(env, policy)
plot_sa_values(env, policy_sa_visitations, title='Q-hat_CQL Visitation')
