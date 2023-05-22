# 🚀 Getting Started with AngoraPy

Welcome to the AngoraPy documentation! This README will guide you through the installation process and provide you with a basic example on how to train an agent. For more advanced examples, please refer to other tutorials in this repository. The main README of the repository contains a list of them.

## Setup and Installation


### Prerequisites
AngoraPy requires Python 3.6 or higher. It is recommended to use a virtual environment to install AngoraPy and its dependencies. Additionally, some prerequisites are required. 

On Ubuntu, these can be installed by running

    sudo apt-get install swig

Additionally, to run AngoraPy with its native distribution, you need MPI installed. On Ubuntu, this can be done by running

    sudo apt-get install libopenmpi-dev

However, any other MPI implementation should work as well.

### Installing AngoraPy

#### Binaries
AngoraPy is available as a binary package on PyPI. To install it, run 

    pip install angorapy

in your terminal.

If you would like to install a specific version, you can specify it by appending `==<version>` to the command above. For example, to install version 0.9.0, run 

    pip install angorapy==0.9.0

#### Source Installation
To install AngoraPy from source, clone the repository and run `pip install -e .` in the root directory.

### Post-Installation

#### MuJoCo
Gym installs both MuJoCo's new native Python bindings and the old mujoco-py bindings. You will not need the latter for AngoraPy, so you may uninstall it by running

    pip uninstall mujoco-py

#### Test Your Installation
You can test your installation by running the following command in your terminal:

    python -m angorapy.train CartPole-v1

To test your MPI installation, run

    mpirun -np <numthreads> --use-hwthread-cpus python -m angorapy.train LunarLanderContinuous-v2

where `<numthreads>` is the number of threads you want to (and can) use.

## Basic Training Script
To train agents with custom models, environments, etc. you write your own script. The following is a minimal example:

```python
from angorapy.common.wrappers import make_env
from angorapy.models import get_model_builder
from angorapy.agent.ppo_agent import PPOAgent

env = make_env("LunarLanderContinuous-v2")
model_builder = get_model_builder("simple", "ffn")
agent = PPOAgent(model_builder, env)
agent.drill(100, 10, 512)
```

This trains a basic agent with a feedforward neural network on the LunarLanderContinuous-v2 environment for 100 cycles, 10 epochs of optimization per cycle and a per-worker horizon of 512. For more details check the Jupyter notebook tutorial in this directory.