import copy
import os
import random
import time
from typing import Optional

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
from os import path
from PIL import Image


class Maze_Builder(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"],
                "video.frames_per_second": 30}

    def __init__(self, grid_size=(25, 25), root='Training_Maze'):
        self.root = root    # 迷宫存储文件夹
        self.grid_size = grid_size
        self.grid_block_num = self.grid_size[0] * self.grid_size[1]     # 用于标定全局block数量
        self.state_transform = dict()   # 用于记录所有状态转移空间
        self.terminate_idx = dict()     # 用于记录所有终止状态
        self.screen = rendering.Viewer(600, 400)
        self.states = range(self.grid_block_num)    # 用于创建所有可遍历状态
        self.state = 0              # 用于记录迷宫中当前位置index
        self.action = ['B', 'R']
        self.maze = None            # 用于从迷宫文件中读取读取数据
        self.flag = None            # 用于创建传向网络的可视化maze标志
        # Testing
        self.maze = np.array(Image.open(os.path.join(self.root, 'maze_0.png')))
        self.flag = self.check_block(self.maze.transpose((2, 0, 1)))
        self.build_state_space()

    def build_state_space(self):
        assert self.grid_size[0] == self.grid_size[1]
        row, col = self.grid_size

        grid_sum = row * col
        for idx in range(grid_sum):
            if (idx // row) == 0 and (idx % row) == (row - 1):      # top right block
                idx_temp = idx + row
                reward_temp = self._reward_calculate(idx + row)
                self.state_transform[f'{idx}_B'] = idx_temp, reward_temp
            elif idx == grid_sum - 1:                               # final block
                continue
            elif (idx // row) == (row - 1):                         # right side blocks
                idx_temp = idx + 1
                reward_temp = self._reward_calculate(idx + 1)
                self.state_transform[f'{idx}_R'] = idx_temp, reward_temp
            elif idx % row == row - 1:                              # bottom side blocks
                idx_temp = idx + row
                reward_temp = self._reward_calculate(idx + row)
                self.state_transform[f'{idx}_B'] = idx_temp, reward_temp
            else:
                idx_bottom_temp = idx + row
                reward_bottom_temp = self._reward_calculate(idx + row)
                idx_right_temp = idx + 1
                reward_right_temp = self._reward_calculate(idx + 1)
                self.state_transform[f'{idx}_B'] = idx_bottom_temp, reward_bottom_temp
                self.state_transform[f'{idx}_R'] = idx_right_temp, reward_right_temp

    def _reward_calculate(self, idx):
        if self.flag[idx]:
            return -100
        elif idx == self.grid_block_num - 1:
            return 100
        else:
            return 0

    @staticmethod
    def check_block(grid_map):
        if not isinstance(grid_map, np.ndarray):
            grid_check = np.array(grid_map)[0, :, :]
        else:
            grid_check = grid_map[0, :, :]
        flag = list(map(lambda x: 0 if x == 255 else 1, grid_check.reshape(-1)))
        flag[0], flag[-1] = 0, 0     # 初始位置不标记
        return np.array(flag)

    def step(self, u):
        state = self.state
        action = self.action[u]
        # 终止条件
        if state in self.terminate_idx:
            return self._get_obs(), 0, True, {}
        key = f"{state}_{action}"

        if key in self.state_transform:
            next_state, reward = self.state_transform[key]
        else:
            next_state, reward = state, 0
        self.state = next_state

        is_terminal = False
        if next_state in self.terminate_idx:
            is_terminal = True

        return self._get_obs(), reward, is_terminal, {}

    def terminate_index_collect(self):
        for idx, val in enumerate(self.flag):
            if val == 1:
                self.terminate_idx[idx] = val

    def reset(self, seed: Optional[int] = None):
        self.state_transform.clear()
        self.terminate_idx.clear()
        maze_file_list = os.listdir(os.path.join(self.root))
        maze_sample = random.sample(maze_file_list, 1)
        self.maze = np.array(Image.open(os.path.join(self.root, maze_sample[0])))
        self.flag = self.check_block(self.maze.transpose((2, 0, 1)))
        self.build_state_space()
        self.terminate_index_collect()
        self.state = 0

    def _get_obs(self):
        img = copy.deepcopy(self.maze)
        img[0, 0, :] = [255, 255, 255]
        img[self.state//self.grid_size[0], self.state % self.grid_size[0], :] = [20, 40, 222]
        return img.transpose((2, 0, 1))

    def render(self, mode='human', close=False):
        line1 = rendering.Line((100, 300), (500, 300))
        line2 = rendering.Line((100, 200), (500, 200))

        line1.set_color(0, 0, 0)
        line2.set_color(0, 0, 0)

        circle = rendering.make_circle(30)
        circle_transform = rendering.Transform(translation=(100, 200))
        circle.add_attr(circle_transform)

        self.screen.add_geom(circle)
        self.screen.add_geom(line1)
        self.screen.add_geom(line2)
        return self.screen.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.screen:
            self.screen.close()
            self.screen = None


if __name__ == '__main__':
    env = Maze_Builder()
    env.reset()
    for i in range(10):
        img, reward, done, _ = env.step(random.randint(0, 1))
        print(f'done: {done}')

    env.close()

