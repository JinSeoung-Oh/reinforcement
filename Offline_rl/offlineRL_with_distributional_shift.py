import numpy as np


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
