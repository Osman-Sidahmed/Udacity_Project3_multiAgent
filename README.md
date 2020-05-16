# Robotic Arm Contiuous Control

![reacher](https://user-images.githubusercontent.com/37901636/82127330-f34b6980-97b2-11ea-8650-cc3497b1d968.gif)

## About the Project
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location.

* Observation space is 33 variables corresponding to position, linear and angular velocities and so.
Action space is 4 variables corresponding to Torque applied to two joints

* The goal of the agent is to maintain its position at the target location for as many time steps as possible.

* The environment is considered solved when the agent gets an average score of **+30 over 100 consecutive episodes**.

* The strategy learnt must get the agent am minimum average score of +13 over 100 consecutive episodes.

## Dependencies & Environment Setup
1. install python 3.6
1. [Clone the DRLND Repository!](https://github.com/udacity/deep-reinforcement-learning#dependencies) and follow the README.md instructions to install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.
1. Download the _ready-built_ Unity Environment:
     1. [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
     2. [Windows 64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
     3.[Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
1. Place the file in the p2_continuous-control/ folder in the DRLND GitHub repository, and unzip (or decompress) the file

## About the code:
* Continuous_Control.ipynb contains the master code where the agent learn and the resulting models are saved.

* ddpg_agent.py contains the definition of the actor-critic architecture of the agent with its methods (act, step, learn), in addition to two utility classes (Ornstein-Uhlenbeck process noise generator & the replay buffer).

* model.py contains the actor-critic nn module architecture defining the number of layers, their dimensions, batch normalization and the activation used at the output layer.
