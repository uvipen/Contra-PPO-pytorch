# [PYTORCH] Proximal Policy Optimization (PPO) for Contra Nes

## Introduction

Here is my python source code for training an agent to play contra nes. By using Proximal Policy Optimization (PPO) algorithm introduced in the paper **Proximal Policy Optimization Algorithms** [paper](https://arxiv.org/abs/1707.06347). 

For your information, PPO is the algorithm proposed by OpenAI and used for training OpenAI Five, which is the first AI to beat the world champions in an esports game. Specifically, The OpenAI Five dispatched a team of casters and ex-pros with MMR rankings in the 99.95th percentile of Dota 2 players in August 2018.

<p align="center">
  <img src="demo/video-1.gif"><br/>
  <i>Sample result</i>
</p>

## Motivation

It has been a while since I have released my A3C implementation ([A3C code](https://github.com/uvipen/Super-mario-bros-A3C-pytorch)) and PPO implementation ([PPO code](https://github.com/uvipen/Super-mario-bros-PPO-pytorch)) for training an agent to play super mario bros. Since PPO outperforms A3C in the number of levels completed, as a next step, I want to see how the former performs in another famous NES game: Contra


## How to use my code

With my code, you can:

* **Train your model** by running `python train.py`. For example: `python train.py --level 1 --lr 1e-4`
* **Test your trained model** by running `python test.py`. For example: `python test.py --level 1`

## Docker

For being convenient, I provide Dockerfile which could be used for running training as well as test phases

Assume that docker image's name is ppo. You only want to use the first gpu. You already clone this repository and cd into it.

Build:

`sudo docker build --network=host -t ppo .`

Run:

`docker run --runtime=nvidia -it --rm --volume="$PWD"/../Contra-PPO-pytorch:/Contra-PPO-pytorch --gpus device=0 ppo`

Then inside docker container, you could simply run **train.py** or **test.py** scripts as mentioned above.

**Note**: There is a bug for rendering when using docker. Therefore, when you train or test by using docker, please comment line `env.render()` on script **src/process.py** for training or **test.py** for test. Then, you will not be able to see the window pop up for visualization anymore. But it is not a big problem, since the training process will still run, and the test process will end up with an output mp4 file for visualization
