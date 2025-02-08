from .orca import compute_orca_reward
from .baselines import compute_dtw_reward, compute_ot_reward, compute_temporal_ot_reward, compute_tracking_with_threshold_reward, compute_even_distribution_reward, compute_final_frame_reward

def load_matching_fn(fn_name, fn_config):
    fn = None 

    if fn_name == "orca":
        fn = lambda cost_matrix: compute_orca_reward(cost_matrix, tau=float(fn_config.get("tau", 1)))
    elif fn_name == "ot":
        fn = lambda cost_matrix: compute_ot_reward(cost_matrix, ent_reg=float(fn_config.get("ent_reg", .01)))
    elif fn_name == "temporal_ot":
        fn = lambda cost_matrix: compute_temporal_ot_reward(cost_matrix, mask_k=float(fn_config.get("mask_k",10)), ent_reg=float(fn_config.get("ent_reg", 0.01)))
    elif fn_name == "dtw":
        fn = compute_dtw_reward
    elif fn_name == "even":
        fn = lambda cost_matrix: compute_even_distribution_reward(cost_matrix, mask_k=float(fn_config.get("mask_k",0)))
    elif fn_name == "final_frame":
        fn = compute_final_frame_reward
    elif fn_name == "threshold":
        fn = lambda cost_matrix: compute_tracking_with_threshold_reward(cost_matrix, threshold=float(fn_config.get("threshold", 0.9)))
    else:
        raise Exception(f"Invalid fn {fn_name}")
    
    return fn