import numpy as np
from .utils import bordered_identity_like, mask_optimal_transport_plan, dtw, dtw_path
import ot

def compute_temporal_ot_reward(cost_matrix,
                                mask_k: int = 10,
                                niter: int = 100,
                                ent_reg: float = 0.01):
    """
    TemporalOT reward, as implemented in (Fu et al., Robot Policy Learning with Temporal Optimal Transport Reward, NeurIPS 2024)
    Code from https://github.com/fuyw/TemporalOT
    """

    # optimal weights 
    mask = bordered_identity_like(cost_matrix.shape[0], cost_matrix.shape[1], k=mask_k)
    transport_plan = mask_optimal_transport_plan(cost_matrix,
                                                mask,
                                                niter,
                                                ent_reg)
    

    ot_cost = np.sum(transport_plan * cost_matrix, axis=1)
    ot_reward = -ot_cost
    return ot_reward, {"assignment": transport_plan}

def compute_ot_reward(cost_matrix, ent_reg=.01) -> np.ndarray:
    """
    Entropy regularized optimal transport reward
    """
    
    # Calculate the OT plan between the reference sequence and the observed sequence
    obs_weight = np.ones(cost_matrix.shape[0]) / cost_matrix.shape[0]
    ref_weight = np.ones(cost_matrix.shape[1]) / cost_matrix.shape[1]

    if ent_reg == 0:
        T = ot.emd(obs_weight, ref_weight, cost_matrix)  # size: (train_freq, ref_seq_len)
    else:
        T = ot.sinkhorn(obs_weight, ref_weight, cost_matrix, reg=ent_reg, log=False)  # size: (train_freq, ref_seq_len)

    # Normalize the path so that each row sums to 1
    normalized_T = T / np.expand_dims(np.sum(T, axis=1), 1)

    # Calculate the OT cost for each timestep
    #   sum by row of (cost matrix * OT plan)
    ot_cost = np.sum(cost_matrix * normalized_T, axis=1)  # size: (train_freq,)

    final_reward = -ot_cost
    return final_reward, {"assignment": normalized_T}

def compute_dtw_reward(cost_matrix):
    """
    Compute the reward with an assignment matrix that uses dynamic time warping
    """
    _, accumulated_cost_matrix = dtw(cost_matrix)
    path = dtw_path(accumulated_cost_matrix)

    # Normalize the path so that each row sums to 1
    normalized_path = path / np.expand_dims(np.sum(path, axis=1), 1)
    dtw_cost = np.sum(cost_matrix * normalized_path, axis=1)  # size: (train_freq,)
    final_reward = -dtw_cost

    return final_reward, {"assignment": normalized_path}

def compute_tracking_with_threshold_reward(cost_matrix, threshold=0.9):
    """
    Compute the reward by estimating progress along the trajectory using a threshold for each subgoal.
    If the soft probability of occupying the current subgoal is above the threshold, we move to the next subgoal.
    
    The final reward is the percent of subgoals completed
    """
    prob_matrix = np.exp(-cost_matrix)
    reward_vector = np.zeros(prob_matrix.shape[0])
    subgoal_tracking_matrix = np.zeros_like(prob_matrix)  # To use the visualization of assignment matrix from other approaches

    curr_subgoal = 0
    total_subgoals = prob_matrix.shape[1]

    for i in range(prob_matrix.shape[0]):
        # 2 components for the reward
        #   - current subgoal reward
        #   - progress reward
        # We then normalize the reward by the total number of subgoals to keep the reward in the range [0, 1]
        reward_vector[i] = (prob_matrix[i, curr_subgoal] + curr_subgoal) / total_subgoals
        subgoal_tracking_matrix[i][curr_subgoal] = 1

        if prob_matrix[i, curr_subgoal] > threshold:
            # Move to the next subgoal until reaching the last subgoal
            curr_subgoal = min(curr_subgoal + 1, prob_matrix.shape[1] - 1)

        # print(f"timestep: {i}; subgoal: {curr_subgoal}/{total_subgoals-1}; reward: {reward_vector[i]}")

    return reward_vector, {"assignment": subgoal_tracking_matrix}

def compute_final_frame_reward(cost_matrix):
    """
    Reward is the distance from the final reference state, ignoring the sequence
    i.e., R = -d(obs, ref[-1])
    """
    assignment = np.zeros_like(cost_matrix)
    assignment[:, -1] = 1

    final_reward = - np.sum(cost_matrix * assignment, axis=1)  # size: (train_freq,)

    return final_reward, assignment

def compute_even_distribution_reward(cost_matrix, mask_k: int=10):
    """
    Compute reward based on an assignment matrix that evenly distributes the frames from obs to ref, with an additional border on each side of size mask_k
    i.e., the first N frames from obs will be distributed to the first frame of ref, and so on, where N is len(obs) // len(ref)
    
    if mask_k == 0 and cost_matrix is square, then this is the identity
    """
    # Calculate the cost matrix between the reference sequence and the observed sequence
    assignment = bordered_identity_like(cost_matrix.shape[0], cost_matrix.shape[1], mask_k)
    normalized_assignment = assignment / np.expand_dims(np.sum(assignment, axis=1), 1)

    even_distributed_cost = np.sum(normalized_assignment * cost_matrix, axis=1)

    final_reward = - even_distributed_cost

    return final_reward, {"assignment": normalized_assignment}

