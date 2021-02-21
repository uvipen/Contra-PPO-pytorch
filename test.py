"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import argparse
import torch
from src.env import create_train_env, ACTION_MAPPING
from src.model import PPO
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Contra Nes""")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output_path", type=str, default="output")
    args = parser.parse_args()
    return args


def test(opt):
    torch.manual_seed(123)
    env = create_train_env(opt.level, "{}/video_{}.mp4".format(opt.output_path, opt.level))
    model = PPO(env.observation_space.shape[0], len(ACTION_MAPPING))
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/ppo_contra_level{}".format(opt.saved_path, opt.level)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/ppo_contra_level{}".format(opt.saved_path, opt.level),
                                         map_location=lambda storage, loc: storage))
    model.eval()
    state = torch.from_numpy(env.reset())
    while True:
        if torch.cuda.is_available():
            state = state.cuda()
        logits, value = model(state)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state)
        env.render()
        if info["level"] > opt.level or done:
            print("Level {} completed".format(opt.level))
            break


if __name__ == "__main__":
    opt = get_args()
    test(opt)
