import numpy as np
import torch 
import multiprocess as mp
from collections.abc import Iterable 
import sys 
sys.path.append("../")
from objective import Objective
from lunar_lander_utils import simulate_lunar_lander_states, get_reward 


class LunarLanderObjective(Objective):
    ''' Lunar Lander optimization task
        Goal is to find a policy for the Lunar Lander 
        smoothly lands on the moon without crashing, 
        thereby maximizing reward 
    ''' 
    def __init__(
        self,
        xs_to_scores_dict={},
        num_calls=0,
        seed=np.arange(50),
        tau=None,
        **kwargs,
    ):
        super().__init__(
            xs_to_scores_dict=xs_to_scores_dict,
            num_calls=num_calls,
            task_id='lunar',
            dim=12,
            lb=0.0,
            ub=1.0,
            **kwargs
        ) 
        self.pool = mp.Pool(mp.cpu_count())
        seed = [seed] if not isinstance(seed, Iterable) else seed 
        self.seed = seed 
        self.dist_func = torch.nn.PairwiseDistance(p=2)


    def query_oracle(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy() 
        x = x.reshape((-1, self.dim)) 
        ns = len(self.seed) 
        rewards = 0.0 
        for seed in range(ns):
            states, m_powers, s_powers, awakes = simulate_lunar_lander_states(
                x, 
                seed, 
            )
            reward = get_reward(seed, states, m_powers, s_powers, awakes)
            rewards += reward 
        mean_reward = rewards/ns 
        return mean_reward 



if __name__ == "__main__":
    obj = LunarLanderObjective()
    x1 = np.array([0.5, 1.0, 0.4, 0.55, 0.5, 1.0, 0.5, 0.5, 0, 0.5, 0.05, 0.05]) * 0.5
    y = obj.query_oracle(x1) 
    import pdb 
    pdb.set_trace() 
