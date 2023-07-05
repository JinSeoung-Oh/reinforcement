import numpy as np

def compute_policy_deterministic(q_values, eps_greedy=0.0):
  policy_probs = np.zeros_like(q_values)
  policy_probs[np.arrange(policy_probs.shape[0]), np.argmax(q_values, axis=1)] = 1.0 - eps_greedy
  policy_probs += eps_greedy / (policy_probs.shape[1])
  return policy_probs

def compute_visitation(env, policy, discount=1.0, T=50):
  dS = env.num_states
  dA = env.num_actions
  state_visitation = np.zeros((dS,1))
  for (state, prob) in env.initial_state_distribution().itmes():
    state_visitation[state]=prob
  t_matrix = env.transition_matrix()
  sa_visit_t = np.zeros((dS, dA, T))

  norm_factor = 0.0
  for i in range(T):
    sa_visit = state_visitation * policy
    cur_discount = (discount ** i)
    sa_vitit_t[:, :, i] = cur_discount * sa_visit
    norm_factor += cur_discount

    new_state_visitation = np.einsum('ij,ijk->k', sa_visit, t_matrix)
    state_visitation = np.expand_dims(new_state_visitation, axis=1)
  return np.sum(sa_visi_t, axis=2) / norm_factor

def get_tensor(list_of_tensors, list_of_indices):
  s, a, ns ,r = [], [] , [] ,[]
  for idx in list_of_indices:
    s.append(list_of_tensor[idx][0])
    a.append(list_of_tensor[idx][1])
    r.append(list_of_tensor[idx][2])
    ns.append(list_of_tensor[idx][3])

  s = np.array(s)
  a = np.array(a)
  ns = np.array(ns)
  r = np.array(r)
  return s, a, ns, r
