#!/bin/bash

# Change to the directory containing the Python script
cd scripts
export CUDA_VISIBLE_DEVICES=1

# Run the Python script  
python3 optimize.py --task_id rover --track_with_wandb True --wandb_entity nmaus --num_initialization_points 1024 --max_n_oracle_calls 20000 --bsz 32 --input_dim 60 --output_dim 256 - run_optimization - done;
python3 optimize.py --task_id rover --track_with_wandb True --wandb_entity nmaus --num_initialization_points 1024 --max_n_oracle_calls 20000 --bsz 32 --input_dim 60 --output_dim 128 - run_optimization - done;
python3 optimize.py --task_id rover --track_with_wandb True --wandb_entity nmaus --num_initialization_points 1024 --max_n_oracle_calls 20000 --bsz 32 --input_dim 60 --output_dim 64 - run_optimization - done;
python3 optimize.py --task_id rover --track_with_wandb True --wandb_entity nmaus --num_initialization_points 1024 --max_n_oracle_calls 20000 --bsz 32 --input_dim 60 --output_dim 32 - run_optimization - done;
python3 optimize.py --task_id rover --track_with_wandb True --wandb_entity nmaus --num_initialization_points 1024 --max_n_oracle_calls 20000 --bsz 32 --input_dim 60 --output_dim 16 - run_optimization - done;
python3 optimize.py --task_id rover --track_with_wandb True --wandb_entity nmaus --num_initialization_points 1024 --max_n_oracle_calls 20000 --bsz 32 --input_dim 60 --output_dim 8 - run_optimization - done;
python3 optimize.py --task_id rover --track_with_wandb True --wandb_entity nmaus --num_initialization_points 1024 --max_n_oracle_calls 20000 --bsz 32 --input_dim 60 --output_dim 4 - run_optimization - done