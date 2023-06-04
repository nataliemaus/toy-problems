import numpy as np
import torch 


class Objective:
    '''Base class for any optimization task
        class supports oracle calls and tracks
        the total number of oracle class made during 
        optimization 
    ''' 
    def __init__(
        self,
        num_calls=0,
        task_id='',
        input_dim=60,
        output_dim=2_000,
        lb=None,
        ub=None,
    ):
        # track total number of times the oracle has been called
        self.num_calls = num_calls
        # string id for optimization task 
        self.task_id = task_id
        # input dim (x)
        self.input_dim = input_dim 
        # output dim (y) 
        self.output_dim = output_dim 
        # absolute upper and lower bounds on search space
        self.lb = lb
        self.ub = ub  


    def __call__(self, xs):
        ''' Input 
                x: Inumeratble type search space points (bsz, input_dim)
            Output
                tensor of scores (bsz, 1)
        '''
        if type(xs) is np.ndarray: 
            xs = torch.from_numpy(xs).float()
        ys = self.xs_to_ys(xs) 
        scores = self.ys_to_scores(ys)
        return scores 


    def xs_to_ys(self, xs):
        ''' Input: 
                Inumeratble type xs: (bsz, input_dim)
            Output:
                torch tensor of ys: (bsz, output_dim)
        '''
        ys = []
        for x in xs:
            ys.append(self.x_to_y(x))
        return torch.cat(ys)

    def ys_to_scores(self, ys):
        ''' Input: 
                Inumeratble type ys: (bsz, output_dim)
            Output:
                torch tensor of ys: (bsz, 1)
        '''
        scores = []
        for y in ys:
            scores.append(self.y_to_score(y))
        return torch.tensor(scores).unsqueeze(-1)


    def x_to_y(self, x): # input dim to output dim 
        ''' Input: 
                a single input space item x
            Output:
                method queries the oracle and returns 
                the corresponding output dim tensor y (1, out_dim)
            Note: 
                This method should also increment self.num_calls 
        '''
        raise NotImplementedError("Must implement x_to_y() specific to desired optimization task")

    
    def y_to_score(self, y): # output dim to score 
        ''' Input: 
                tensor y (1, out_dim)
            Output:
                method queries the oracle and returns 
                the corresponding score y,
        '''
        raise NotImplementedError("Must implement y_to_score() specific to desired optimization task")
