# Rotate Suite

This package implements a series of continuous control problems with continuous and discrete symmetries. 
The environments are base on [DeepMind Control Suite](https://github.com/deepmind/dm_control). 

Rotate Suite is a series of visual control tasks with the goal of rotating a 3D object along its axes to achieve a
goal orientation. Symmetries of the environment are declared by symmetries of the object. Thus, some environments have 
discrete symmetries, while others have continuous symmetries.

![Rotate Suite](figures/env.png)

If you use our code, please cite our [paper](https://arxiv.org/abs/2305.05666): 

```bib
@article{panangaden2023policy,
  title={Policy Gradient Methods in the Presence of Symmetries and State Abstractions},
  author={Panangaden, Prakash and Rezaei-Shoshtari, Sahand and Zhao, Rosie and Meger, David and Precup, Doina},
  journal={arXiv preprint arXiv:2305.05666},
  year={2023}
}
```

## Setup
* Install the following libraries needed for Mujoco and DeepMind Control Suite:
```commandline
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```
* We recommend using a conda virtual environment to run the code.
Create the virtual environment:
```commandline
conda create -n rotate_env python=3.9
conda activate rotate_env
pip install --upgrade pip
```
* Install [Mujoco](https://github.com/deepmind/mujoco) and [DeepMind Control Suite](https://github.com/deepmind/dm_control)
following the official instructions.
* Clone this package and install its dependencies:
```commandline
pip install -r requirements.txt
```
* Finally install the `contextual_control_suite` package with `pip`: 
```commandline
pip install -e .
```

## Instructions
* A demo script showing how to use the contexts is available [here](rotate_suite/demos/demo_env.py).
