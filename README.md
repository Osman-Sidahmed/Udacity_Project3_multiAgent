# Tennis

![tennis](https://user-images.githubusercontent.com/37901636/82143613-6f8e8d00-9845-11ea-98a6-11b76269d6c4.png)

## About the Project
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.

* Observation space is 8 variables corresponding the position and velocity of the ball and racket.
* Action space is 2 variables corresponding to moving towards the net and the other jumping.

* The goal of the agent is to maintain its position at the target location for as many time steps as possible.

* The environment is considered solved when the agent gets an average score of **+0.5 over 100 consecutive episodes where the episode score is the maximum between the two agent's scores**.

## Dependencies & Environment Setup
1. install python 3.6
1. [Clone the DRLND Repository!](https://github.com/udacity/deep-reinforcement-learning#dependencies) and follow the README.md instructions to install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.
1. Download the _ready-built_ Unity Environment:
     1. [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
     2. [Windows 64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
     3.[Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
1. Place the file in the p3_collab-compet/ folder in the DRLND GitHub repository, and unzip (or decompress) the file

## About the code:
* Tennis.ipynb contains the master code where the agent learn and the resulting models are saved.

* ddpg_agent.py contains the definition of multi-agent setup: the multi-agent calss containing two identical agents, Agent class used by the multi-agent class to define the actor-critic structure of each agent, the OUNoise class and the replay buffer.

* model.py contains the actor-critic nn module architecture defining the number of layers, their dimensions, batch normalization and the activation used at the output layer.
