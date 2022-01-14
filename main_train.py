# -*- coding: utf-8 -*-
# @Time    : 2022/1/10 15:38
# @Author  : Scotty
# @FileName: main_train.py
# @Software: PyCharm
import sys
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from utils_tools.DDQN import DDQN
from utils_tools.initialize import maze_build
from utils_tools.Env import Maze_Builder
from tqdm import tqdm

EPOCHS = 5000
MAX_STEP_LENGTH = 120

if __name__ == '__main__':
    env = Maze_Builder()
    grad_size = (3, 25, 25)
    action_size = [0, 1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = DDQN(grad_size, action_size, device)

    epochs_tqdm = tqdm(range(EPOCHS))
    total_reward = []
    for epoch in epochs_tqdm:
        s_t = env.reset()

        reward_hity = 0
        step_tqdm = tqdm(range(MAX_STEP_LENGTH))
        for step in step_tqdm:
            action = agent.get_action(s_t)
            s_t1, reward, done, _ = env.step(action)
            agent.exp_memory(s_t, action, reward, s_t1, done)
            reward_hity += reward
            step_tqdm.set_description(f"epoch:{epoch}, \'TTL\':{agent.t}, step:{step}/{MAX_STEP_LENGTH}, reward:{reward}, done?:{done}, state:{env.state}, loss:{agent.train_loss}")
            if agent.train:
                agent.training()

            if done:
                print('!!!terminated by step error, new env initializing!!!')
                break

            s_t = s_t1
            agent.t += 1
        total_reward.append(reward_hity)
        agent.update_target_model()
        agent.ep += 1

    env.close()
    sys.exit()


