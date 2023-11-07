## from https://towardsai.net/p/machine-learning/policy-gradient-algorithms-mathematics-explained-with-pytorch-implementation
# RL algorithms can be generally categorized into two groups i.e., value-based and policy-based methods. 
# 1. Value-based methods aim at estimating the expected return of the states and selecting an action in that 
#    state which results in the highest expected value, which is rather an indirect way of behaving optimally in an MDP environment. 
# 2. In contrast, policy-based methods try to learn and optimize a policy function, which is basically a mapping from states to actions.
# Policy Gradient (PG) methods are algorithms that aim to learn the optimal policy function directly in a Markov Decision Processes setting (S, A, P, R, γ)
# In PG, the policy π is represented by a parametric function (e.g., a neural network), 
# so we can control its outputs by changing its parameters. The policy π maps state to actions (or probability distributions over actions)
# The goal of PG is to achieve a policy that maximizes the expected cumulative rewards over a trajectory of states and actions

# pi_septa(a|s)
# In the case of Deep RL, we consider this function to be approximated by a Neural Network (with parameter set θ) 
# that takes states (observations) as input and outputs the distribution over actions
# It could be a discrete distribution for discrete actions or a continuous distribution for continuous actions.

# In general, the agent’s goal would be to obtain the maximum cumulative reward over a trajectory of interactions
# R(τ) = sigma (t=0 to H) R(s_t,a_t) = r_0 + r_1 + ... + r_H
# τ is a random variable (R(τ) to be a stochastic function )

# Goal : maximize its expectation, meaning to maximize it on average case, while taking actions with policy π: E[ R(τ); π ]
# E[ R(τ); π_cepth] = E[sigma (t=0 to H) R(s_t, a_t);π_cepth]
