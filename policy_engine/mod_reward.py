
import torch

def make_reward_sparse(env,reward,flag_sparse,threshold_sparcity,initial_pose):
    flag_absorbing_state=False
    if flag_sparse is True:
        if (abs(env.env.data.qpos[0]-initial_pose)>=threshold_sparcity):
            reward = reward
            # flag_absorbing_state=True
        else:
            reward = 0
    return reward,flag_absorbing_state
