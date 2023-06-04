import sys 
sys.path.append("../")
import torch
import random
import numpy as np
import fire
import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import os
os.environ["WANDB_SILENT"] = "True"
import signal 
import gpytorch 
from utils.bo_utils.turbo import (
    TurboState, 
    update_state_unconstrained, 
    generate_batch,
    generate_batch_intermediate_output,
)
from utils.utils import update_surrogate_models, update_surr_model
from utils.bo_utils.ppgpr import GPModelDKL
from gpytorch.mlls import PredictiveLogLikelihood 
from tasks.rover.rover_objective import RoverObjective
try:
    import wandb
    WANDB_IMPORTED_SUCCESSFULLY = True
except ModuleNotFoundError:
    WANDB_IMPORTED_SUCCESSFULLY = False

TASK_ID_TO_OBJECTIVE = {}
TASK_ID_TO_OBJECTIVE['rover'] = RoverObjective

class Optimize(object):
    """
    Optimize high-dim input and output task 
    Args:
        task_id: String id for optimization task, by default the wandb project name will be f'{task_id}-optimize'
        seed: Random seed to be set. If None, no particular random seed is set
        track_with_wandb: if True, run progress will be tracked using Weights and Biases API
        wandb_entity: Username for your wandb account (valid username necessary iff track_with_wandb is True)
        wandb_project_name: Name of wandb project where results will be logged (if no name is specified, default project name will be f"optimimze-{self.task_id}")
        max_n_oracle_calls: Max number of oracle calls allowed (budget). Optimization run terminates when this budget is exceeded
        learning_rte: Learning rate for model updates
        acq_func: Acquisition function, must be either ei or ts (ei-->Expected Imporvement, ts-->Thompson Sampling)
        bsz: Acquisition batch size
        num_initialization_points: Number evaluated data points used to optimization initialize run
        init_n_epochs: Number of epochs to train the surrogate model for on initial data before optimization begins
        num_update_epochs: Number of epochs to update the model(s) for on each optimization step
        print_freq: If verbose, program will print out an update every print_freq steps during optimzation. 
        input_dim: dim of search space for optimization task 
        output_dim: dim of intermediate output space we want to model with GPs
        gp_hidden_dims: tuple giving hidden dims for GP Deep Kernel, if None will be (input_dim, input_dim) by default
        model_intermeidate_output: boolean, if True we use multiple GP models to predict the intermediate output values
    """
    def __init__(
        self,
        task_id: str="rover",
        seed: int=None,
        track_with_wandb: bool=True,
        wandb_entity: str="nmaus",
        wandb_project_name: str="",
        max_n_oracle_calls: int=200_000,
        learning_rte: float=0.001,
        acq_func: str="ts",
        bsz: int=10,
        num_initialization_points: int=100,
        init_n_epochs: int=30,
        num_update_epochs: int=5,
        print_freq: int=10,
        verbose: bool=True,
        input_dim: int=60,
        output_dim: int=20,
        gp_hidden_dims: tuple=None,
        model_intermeidate_output: bool=False,
        max_lookback: int=1_000, # max N train data points to update on each iteration 
    ):
        # add all local args to method args dict to be logged by wandb
        signal.signal(signal.SIGINT, self.handler)
        self.method_args = {}
        self.method_args['init'] = locals()
        del self.method_args['init']['self']
        self.seed = seed
        self.track_with_wandb = track_with_wandb
        self.wandb_entity = wandb_entity 
        self.task_id = task_id
        self.max_n_oracle_calls = max_n_oracle_calls
        self.verbose = verbose
        self.num_initialization_points = num_initialization_points
        self.print_freq = print_freq
        self.input_dim = input_dim 
        self.output_dim = output_dim 
        self.bsz = bsz 
        self.init_n_epochs = init_n_epochs
        self.num_update_epochs = num_update_epochs
        self.initial_model_training_complete = False
        self.learning_rte = learning_rte
        if gp_hidden_dims is None:
            gp_hidden_dims = (self.input_dim // 2, self.input_dim // 2)
        self.gp_hidden_dims = gp_hidden_dims
        self.acq_func = acq_func
        self.model_intermeidate_output = model_intermeidate_output
        self.max_lookback = max_lookback
        
        self.set_seed()
        if wandb_project_name: # if project name specified
            self.wandb_project_name = wandb_project_name
        else: # otherwise use defualt
            self.wandb_project_name = f"{self.task_id}-optimization"
        if not WANDB_IMPORTED_SUCCESSFULLY:
            assert not self.track_with_wandb, "Failed to import wandb, to track with wandb, try pip install wandb"
        if self.track_with_wandb:
            assert self.wandb_entity, "Must specify a valid wandb account username (wandb_entity) to run with wandb tracking"

        # initialize latent space objective (self.objective) for particular task
        self.initialize_objective()
        # initialize train data for particular task
        self.load_train_data()
        # check for correct initialization of train data:
        assert torch.is_tensor(self.train_y), \
            f"load_train_data() must set self.train_y to a torch tensor of ys,\
            instead got self.train_y of type {type(self.train_y)}" 
        assert torch.is_tensor(self.train_scores), \
            f"load_train_data() must set self.train_scores to a torch tensor of scores,\
            instead got self.train_scores of type {type(self.train_scores)}" 
        assert (self.train_y.shape[0] == self.num_initialization_points) and \
            (self.train_y.shape[1] == self.output_dim), \
            f"load_train_data() must initialize self.train_x with dims \
            (self.num_initialization_points,self.input_dim), instead got \
            self.train_x with dims {self.train_x.shape}"  
        assert (self.train_x.shape[0] == self.num_initialization_points) and \
            (self.train_x.shape[1] == self.input_dim), \
            f"load_train_data() must initialize self.train_x with dims \
            (self.num_initialization_points, self.input_dim), instead got \
            self.train_x with dims {self.train_x.shape}"
        assert (self.train_scores.shape[0] == self.num_initialization_points) and \
            (self.train_scores.shape[1] == 1), \
            f"load_train_data() must initialize self.train_scores with dims \
            (self.num_initialization_points, 1), instead got \
            self.train_scores with dims {self.train_scores.shape}"


    def handler(self, signum, frame):
        # If we Ctrl-c, make sure we terminate wandb tracker 
        if self.track_with_wandb:
            print("Ctrl-c hass been pressed, now terminating wandb tracker...")
            self.tracker.finish() 
            msg = "Wandb tracker terminated, now exiting..."
            print(msg, end="", flush=True)
        exit(1)


    def initialize_objective(self):
        ''' Initialize Objective for specific task
            must define self.objective object
        '''
        self.objective = TASK_ID_TO_OBJECTIVE[self.task_id](
            num_calls=0,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        return self


    def load_train_data(self):
        ''' Load in or randomly initialize self.num_initialization_points
            total initial data points to kick-off optimization 
        '''
        self.train_x = torch.rand(self.num_initialization_points, self.input_dim) # 0 to 1 uniform 
        if (self.objective.lb is not None) and (self.objective.ub is not None): # convert to random uniform in bounds if bounds given 
            self.train_x = self.train_x*(self.objective.ub - self.objective.lb) + self.objective.lb 
        
        self.train_y = self.objective.xs_to_ys(self.train_x)
        self.train_scores = self.objective.ys_to_scores(self.train_y)
        return self
    

    def set_seed(self):
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = False
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = False
        if self.seed is not None:
            torch.manual_seed(self.seed) 
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            os.environ["PYTHONHASHSEED"] = str(self.seed)
        return self


    def create_wandb_tracker(self):
        if self.track_with_wandb:
            self.tracker = wandb.init(
                project=self.wandb_project_name,
                entity=self.wandb_entity,
                config={k: v for method_dict in self.method_args.values() for k, v in method_dict.items()},
            ) 
            self.wandb_run_name = wandb.run.name
        else:
            self.tracker = None 
            self.wandb_run_name = 'no-wandb-tracking'
        return self


    def update_best(self,):
        self.best_score = self.train_scores.max() 
        self.best_x = self.train_x[self.train_scores.argmax()]
        self.best_y = self.train_y[self.train_scores.argmax()]
        return self 

    def log_data_to_wandb_on_each_loop(self):
        if self.track_with_wandb:
            dict_log = {
                "n_oracle_calls":self.objective.num_calls,
                "best_x":self.best_x,
                "best_y":self.best_y,
                "best_score":self.best_score,
                "tr_length":self.tr_state.length
            }
            self.tracker.log(dict_log)
        return self 

    def initialize_trust_region(self):
        self.tr_state = TurboState( 
            dim=self.objective.input_dim,
            batch_size=self.bsz, 
        )
        return self 

    def run_optimization(self): 
        ''' Main optimization loop
        '''
        # creates wandb tracker iff self.track_with_wandb == True
        self.create_wandb_tracker()
        # initialize trust reigon 
        self.initialize_trust_region() 
        # grab best point found so far (in init data)
        self.update_best()
        # log init data
        self.log_data_to_wandb_on_each_loop()
        # initialize surrogate model(s)
        self.initialize_surrogate_model() 
        #main optimization loop 
        self.step_num = 0
        while self.objective.num_calls < self.max_n_oracle_calls:
            # update surrogate model(s) on data 
            self.update_surrogate_model()
            # generate new candidate points, evaluate them, and update data
            self.acquisition() 
            # check if restart is triggered for trust reggion and restart it as needed
            self.restart_tr_as_needed() 
            # update best point found so far for logging 
            self.update_best()
            # log data to wandb 
            self.log_data_to_wandb_on_each_loop()
            # periodically print updates
            self.step_num += 1
            if self.verbose and (self.step_num % self.print_freq == 0):
                self.print_progress_update()

        # if verbose, print final results
        if self.verbose:
            print("\nOptimization Run Finished, Final Results:")
            self.print_progress_update()

        # terminate wandb tracker 
        if self.track_with_wandb:
            self.tracker.finish()

        return self 

    def initialize_surrogate_model(self ):
        n_pts = min(self.train_x.shape[0], 1024) 
        if self.model_intermeidate_output:
            self.model = None
            self.mll = None 
            self.models_list = []
            self.mlls_list = []
            for _ in range(self.output_dim):
                likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda() 
                model = GPModelDKL(
                    self.train_x[:n_pts, :].cuda(), 
                    likelihood=likelihood,
                    hidden_dims=self.gp_hidden_dims, 
                ).cuda() 
                mll = PredictiveLogLikelihood(
                    model.likelihood, 
                    model, 
                    num_data=self.train_x.size(-2)
                )
                model = model.eval() 
                model = model.cuda() 
                self.models_list.append(model)
                self.mlls_list.append(mll)
        else:
            self.models_list = None
            self.mlls_list = None 
            likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda() 
            self.model = GPModelDKL(
                self.train_x[:n_pts, :].cuda(), 
                likelihood=likelihood,
                hidden_dims=self.gp_hidden_dims, 
            ).cuda() 
            self.mll = PredictiveLogLikelihood(
                self.model.likelihood, 
                self.model, 
                num_data=self.train_x.size(-2)
            ) 
            self.model = self.model.eval() 
            self.model = self.model.cuda() 
            
        return self 


    def update_surrogate_model(self ): 
        if not self.initial_model_training_complete:
            # first time training surr model --> train on all data
            n_epochs = self.init_n_epochs
            X = self.train_x
            Y = self.train_y 
            S = self.train_scores.squeeze(-1)
        else:
            # otherwise, only train on most recent batch of data
            lookback = min(self.max_lookback, len(self.train_x))
            lookback = max(lookback, self.bsz)
            n_epochs = self.num_update_epochs
            X = self.train_x[-lookback:]
            Y = self.train_y[-lookback:] 
            S = self.train_scores[-lookback:].squeeze(-1)

        if self.model_intermeidate_output: 
            self.models_list = update_surrogate_models(
                models_list=self.models_list,
                mlls_list=self.mlls_list,
                learning_rte=self.learning_rte,
                train_x=X,
                train_y=Y,
                n_epochs=n_epochs,
            ) 
        else:
            self.model = update_surr_model(
                model=self.model,
                mll=self.mll,
                learning_rte=self.learning_rte,
                train_x=X,
                train_y=S,
                n_epochs=n_epochs,
            )
        
        self.initial_model_training_complete = True
        return self 


    def print_progress_update(self):
        ''' Update printed each periodically during optimization
            (only used if self.verbose==True)
            More print statements can be added here as desired
        '''
        print(f"\nOptimization progress on {self.task_id}")
        print(f"    Number of optimization steps completed: {self.step_num}")
        print(f"    Total Number of Oracle Calls (Function Evaluations): {self.objective.num_calls}")
        print(f"    Best score found: {self.best_score}") 
        print(f"    Trust region length: {self.tr_state.length}") 
        if self.track_with_wandb:
            print(f"    See more updates in wandb run {self.wandb_project_name}, {wandb.run.name}")
        return self

    def restart_tr_as_needed(self):
        if self.tr_state.restart_triggered:
            self.tr_state = TurboState( 
                dim=self.objective.input_dim,
                batch_size=self.bsz, 
            )
        return self 


    def acquisition(self):   
        '''Generate new candidate points,
        evaluate them, and update data
        '''
        if self.objective.lb is None:  # if no bounds 
            absolute_bounds = None 
        else:
            absolute_bounds=(self.objective.lb, self.objective.ub)
        if self.model_intermeidate_output: 
            x_next = generate_batch_intermediate_output(
                state=self.tr_state,
                models_list=self.models_list,
                objective=self.objective,
                X=self.train_x,
                Y=self.train_y,
                S=self.train_scores,
                batch_size=self.bsz, 
                acqf=self.acq_func,
                absolute_bounds=absolute_bounds,
            ) 
        else:
            x_next = generate_batch(
                state=self.tr_state,
                model=self.model, # GP model
                X=self.train_x,  # Evaluated points on the domain [0, 1]^d
                Y=self.train_scores,  # Function values
                batch_size=self.bsz,
                acqf=self.acq_func,  # "ei" or "ts"
                absolute_bounds=absolute_bounds, 
            )
        if len(x_next.shape) == 1:
            x_next = x_next.unsqueeze(0)
        y_next = self.objective.xs_to_ys(x_next) 
        s_next = self.objective.ys_to_scores(y_next)
        self.tr_state = update_state_unconstrained(self.tr_state, s_next)
        self.train_x = torch.cat((self.train_x, x_next.detach().cpu() ), dim=-2)
        self.train_y = torch.cat((self.train_y, y_next.detach().cpu() ), dim=-2)
        self.train_scores = torch.cat((self.train_scores, s_next.detach().cpu() ), dim=-2) 

        return self 
    

    def done(self):
        return None


def new(**kwargs):
    return Optimize(**kwargs)

if __name__ == "__main__":
    fire.Fire(Optimize)
    # python3 optimize.py 
