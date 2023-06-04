# toy-problems

docker run -v /home/nmaus/toy-problems:/workspace/toy-problems --gpus all -it nmaus/meta:latest

cd scripts 

CUDA_VISIBLE_DEVICES=0 python3 optimize.py --task_id rover --track_with_wandb True --wandb_entity nmaus --num_initialization_points 1024 --max_n_oracle_calls 20000 --bsz 32 --input_dim 60 --output_dim 16 --model_intermeidate_output True - run_optimization - done 


# regular opt... 
CUDA_VISIBLE_DEVICES=1 python3 optimize.py --task_id rover --track_with_wandb True --wandb_entity nmaus --num_initialization_points 1024 --max_n_oracle_calls 20000 --bsz 32 --input_dim 60 --output_dim 2000 - run_optimization - done 


chmod +x run_script.sh
./run_script.sh

chmod +x run_script2.sh
./run_script2.sh 