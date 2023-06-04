import torch 
import sys 
sys.path.append("../../") 
from tasks.objective import Objective
from tasks.rover.rover_utils import create_large_domain


class RoverObjective(Objective):
    ''' Rover optimization task
        Goal is to find a policy for the Rover which
        results in a trajectory that moves the rover from
        start point to end point while avoiding the obstacles,
        thereby maximizing reward 
    ''' 
    def __init__(
        self,
        input_dim=60,
        output_dim=1_000,
        num_calls=0,
        f_max=5.0, # default 
        **kwargs,
    ):
        assert input_dim % 2 == 0 # input dim must be divisible by 2
        lb = -0.5 * 4 / input_dim 
        ub = 4 / input_dim 
        assert output_dim % 2 == 0 # output dim must be divisible by 2

        # create rover domain 
        self.domain = create_large_domain(
            n_points=input_dim // 2,
            n_samples=output_dim // 2,
        )
        self.offset = f_max 
        # rover oracle needs torch.double datatype 
        self.tkwargs={"dtype": torch.double}

        super().__init__(
            num_calls=num_calls,
            task_id='rover',
            input_dim=input_dim,
            output_dim=output_dim,
            lb=lb,
            ub=ub,
            **kwargs,
        ) 

    def x_to_y(self, x):
        ''' Input: 
                policy x (1, input_dim)
            Output:
                corresponding trajectory tensor y (1, output_dim)
        '''
        traj_points = self.policy_to_trajectory(x)
        traj_tensor = torch.from_numpy(traj_points).float() 
        traj_tensor = traj_tensor.reshape(1, -1)
        self.num_calls += 1
        return traj_tensor 
    
    def y_to_score(self, y):
        ''' Input: 
                tensor y (output_dim,) giving trajectory of Rover 
            Output:
                corresponding reward value (single number)
        '''
        traj_points = y.reshape(-1, 2) # (out_dim,) --> (out_dim//2, 2)
        traj_points = traj_points.to(**self.tkwargs) 
        traj_points = traj_points.cpu().numpy() 
        reward = self.trajectory_to_reward(traj_points) 
        return reward 

    def policy_to_trajectory(self, policy):
        traj_points = self.domain.trajectory(policy.cpu().numpy()) 
        # traj_points.shape = (1000, 2) = (output_dim//2, 2) 
        return traj_points 

    def trajectory_to_reward(self, traj_points):
        cost = self.domain.traj_to_cost(traj_points)
        reward = -1 * cost 
        reward = reward + self.offset
        reward_torch = torch.tensor(reward).to(**self.tkwargs) 
        return reward_torch.item() 


if __name__ == "__main__":
    input_dim = 10 # must be divisible by 2 
    output_dim = 100 # must be divisible by 2 
    bsz = 10
    objective = RoverObjective(
        input_dim=input_dim,
        output_dim=output_dim,
    )

    xs = torch.randn(bsz,input_dim)
    # scores = objective(xs)

    ys = objective.xs_to_ys(xs)  # tensor (bsz, output_dim)
    scores = objective.ys_to_scores(ys) # tensor (bsz, 1)

