FROM pytorch/pytorch:latest 
# copy data to path in local computer from path in docker image (data/)
RUN pip install fire 
RUN pip install rdkit-pypi
RUN pip install gpytorch 
RUN pip install botorch
RUN pip install wandb
RUN pip install pytorch_lightning
RUN pip install pandas
RUN pip install networkx
RUN pip install matplotlib
RUN pip install sklearn
RUN apt update
RUN apt install -y build-essential
RUN apt install -y libxrender1 libxext6 software-properties-common apt-utils
RUN conda install -y pomegranate
RUN pip install fcd-torch
# Set up gym and mujoco so we can run Hopper/Swimmer etc 
RUN pip install free-mujoco-py
RUN pip install gym
RUN apt update
RUN apt install -y build-essential
RUN apt install -y libosmesa6-dev
RUN apt install -y libgl-dev
# # Compile mujoco_py stuff so it doesn't happen from scratch on each docker run 
# RUN python -c "import mujoco_py"
# for saving mujoco videos: 
RUN conda install -c conda-forge ffmpeg 
ENV WANDB_API_KEY=dfa956c5bfb9fa492ebf9adede99093772919518

# Lunar 
RUN apt-get autoclean 
RUN apt-get clean
RUN apt-get update -y
RUN apt-get install -y swig 
RUN pip install box2d-py
RUN pip install box2d 
RUN pip install multiprocess

# random 
RUN pip install pyarrow 

# docker build -t nmaus/meta .
# docker push nmaus/meta 
# docker pull nmaus/meta
# docker run -v /home/nmaus/toy-problems:/workspace/toy-problems --gpus all -it nmaus/meta:latest

