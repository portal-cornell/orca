import numpy as np

def compute_orca_reward(cost_matrix, tau=1):

    """
    # max of orca in previous learner timestep, orca in current learner timestep up to previous state and occupying current state
    covered[t, t'] = max(covered[t-1, t'], covered[t, t'-1] * exp(-cost(t, t')) ) 
    covered[:, -1] = 1 # ref -1 is always covered
    covered[-1, :] = 0 # nothing has been covered by learner frame -1
    """
    prob_matrix = np.exp(-cost_matrix / tau)

    covered = np.zeros_like(prob_matrix)
    covered[0,0] = prob_matrix[0,0]

    for i in range(1, covered.shape[0]):
        covered[i, 0] = max(covered[i-1, 0], prob_matrix[i, 0])

    for j in range(1, covered.shape[1]):
        covered[0, j] = covered[0, j-1] * prob_matrix[0, j]

    for i in range(1, covered.shape[0]):
        for j in range(1, covered.shape[1] - 1):
            covered[i,j] = max(covered[i-1, j], covered[i, j-1] * prob_matrix[i, j])
    
    covered[:, -1] = covered[:, -2] * prob_matrix[:, -1]

    # final_reward = orca_scaled_probabilities.sum(axis=1)
    final_reward = covered[:, -1]
    return final_reward, {"assignment": covered}
