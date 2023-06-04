#!/bin/bash

# Change to the directory containing the Python script
cd scripts
export CUDA_VISIBLE_DEVICES=0

# Run the Python script 
python3 optimize.py --task_id rover --track_with_wandb True --wandb_entity nmaus --num_initialization_points 1024 --max_n_oracle_calls 20000 --bsz 32 --input_dim 60 --output_dim 4 --model_intermeidate_output True - run_optimization - done;
python3 optimize.py --task_id rover --track_with_wandb True --wandb_entity nmaus --num_initialization_points 1024 --max_n_oracle_calls 20000 --bsz 32 --input_dim 60 --output_dim 8 --model_intermeidate_output True - run_optimization - done;
python3 optimize.py --task_id rover --track_with_wandb True --wandb_entity nmaus --num_initialization_points 1024 --max_n_oracle_calls 20000 --bsz 32 --input_dim 60 --output_dim 16 --model_intermeidate_output True - run_optimization - done;
python3 optimize.py --task_id rover --track_with_wandb True --wandb_entity nmaus --num_initialization_points 1024 --max_n_oracle_calls 20000 --bsz 32 --input_dim 60 --output_dim 32 --model_intermeidate_output True - run_optimization - done;
python3 optimize.py --task_id rover --track_with_wandb True --wandb_entity nmaus --num_initialization_points 1024 --max_n_oracle_calls 20000 --bsz 32 --input_dim 60 --output_dim 64 --model_intermeidate_output True - run_optimization - done;
python3 optimize.py --task_id rover --track_with_wandb True --wandb_entity nmaus --num_initialization_points 1024 --max_n_oracle_calls 20000 --bsz 32 --input_dim 60 --output_dim 128 --model_intermeidate_output True - run_optimization - done;
python3 optimize.py --task_id rover --track_with_wandb True --wandb_entity nmaus --num_initialization_points 1024 --max_n_oracle_calls 20000 --bsz 32 --input_dim 60 --output_dim 512 --model_intermeidate_output True - run_optimization - done;
python3 optimize.py --task_id rover --track_with_wandb True --wandb_entity nmaus --num_initialization_points 1024 --max_n_oracle_calls 20000 --bsz 32 --input_dim 60 --output_dim 1024 --model_intermeidate_output True - run_optimization - done;
