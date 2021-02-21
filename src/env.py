"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import retro
from gym.spaces import Box
from gym import Wrapper
import cv2
import numpy as np
import subprocess as sp
import torch.multiprocessing as mp

ACTION_MAPPING = {
    0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Fire
    1: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Left + fire
    2: [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Right + fire
    3: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Down + fire
    4: [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Jump + fire

    5: [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # Right + up + fire
    6: [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # Right + down + fire
    7: [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  # Right + jump + fire
}


class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "60", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())


def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))


class CustomReward(Wrapper):
    def __init__(self, env=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_pos = 0
        self.curr_score = 0
        self.curr_lives = 2
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def step(self, action):
        state, _, done, info = self.env.step(action)
        if self.monitor:
            self.monitor.record(state)
        n_state, _, _, _ = self.env.step([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if self.monitor:
            self.monitor.record(n_state)
        state = process_frame(state)
        reward = min(max((info["xscroll"] - self.curr_pos - 0.01), -3), 3)
        self.curr_pos = info["xscroll"]
        reward += min(max((info["score"] - self.curr_score), 0), 2)
        self.curr_score = info["score"]
        if info["lives"] < self.curr_lives:
            reward -= 15
            self.curr_lives = info["lives"]
        if done:
            if info["lives"] != 0:
                reward += 50
            else:
                reward -= 35
        return state, reward / 10., done, info

    def reset(self):
        self.curr_pos = 0
        self.curr_score = 0
        self.curr_lives = 2
        return process_frame(self.env.reset())


class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action):
        total_reward = 0
        last_states = []
        for i in range(self.skip):
            state, reward, done, info = self.env.step(ACTION_MAPPING[action])
            total_reward += reward
            if i >= self.skip / 2:
                last_states.append(state)
            if done:
                self.reset()
                return self.states[None, :, :, :].astype(np.float32), total_reward, done, info
        max_state = np.max(np.concatenate(last_states, 0), 0)
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state
        return self.states[None, :, :, :].astype(np.float32), total_reward, done, info

    def reset(self):
        state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.skip)], 0)
        return self.states[None, :, :, :].astype(np.float32)


def create_train_env(level, output_path=None):
    env = retro.make("Contra-Nes", state="Level{}".format(level), use_restricted_actions=retro.Actions.FILTERED)
    if output_path:
        monitor = Monitor(240, 224, output_path)
    else:
        monitor = None

    env = CustomReward(env, monitor)
    env = CustomSkipFrame(env)
    return env


class MultipleEnvironments:
    def __init__(self, level, num_envs, output_path=None):
        self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        env = create_train_env(level, output_path=output_path)
        self.num_states = env.observation_space.shape[0]
        env.close()
        self.num_actions = len(ACTION_MAPPING)
        for index in range(num_envs):
            process = mp.Process(target=self.run, args=(index, level, output_path))
            process.start()
            self.env_conns[index].close()

    def run(self, index, level, output_path):
        env = create_train_env(level, output_path=output_path)
        self.agent_conns[index].close()
        while True:
            request, action = self.env_conns[index].recv()
            if request == "step":
                self.env_conns[index].send(env.step(action.item()))
            elif request == "reset":
                self.env_conns[index].send(env.reset())
            else:
                raise NotImplementedError
