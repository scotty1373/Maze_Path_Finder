# -*- coding: utf-8 -*-
# @Time    : 2022/1/8 17:35
# @Author  : Scotty
# @FileName: initialize.py
# @Software: PyCharm
import numpy
import numpy as np
from PIL import Image
import random
import shutil
import os
import copy
from maze_build import Checking

NUM_BLOCKS = 75
GRID_SIZE = [3, 25, 25]
TOTAL_MAP = 30
POINT_FLAG = []


class maze_build:
    def __init__(self, num_block, total_map, grid_size):
        self.grad_size = grid_size
        self.num_block = num_block
        self.total_map = total_map
        self.limit = self.grad_size[1] * self.grad_size[2]
        self.maze_cache = np.zeros([self.total_map] + self.grad_size)
        self.maze_flag = np.zeros([self.total_map] + [self.limit])

    def maze_make(self):
        grid_map = np.full_like(np.zeros(self.grad_size), 255)
        grid_map, flag = self.assign_block(grid_map)
        return np.uint8(grid_map), flag

    def check_reward(self, flag):
        reward_point = []
        for idx in range(self.limit):
            if flag[idx] == 1:
                reward_point.append(-100)
            elif idx == self.limit - 1:
                reward_point.append(100)
            else:
                reward_point.append(0)
        return np.array(reward_point)

    def check_block(self, grid_map):
        if not isinstance(grid_map, numpy.ndarray):
            grid_check = np.array(grid_map)[0, :, :]
        else:
            grid_check = grid_map[0, :, :]
        flag = list(map(lambda x: 0 if x > 0 else 1, grid_check.reshape(-1)))
        return np.array(flag)

    def assign_block(self, grid_map):
        flag = np.zeros(self.limit)
        grid_test = copy.deepcopy(grid_map)
        marked_block = random.sample(range(1, self.limit - 1), self.num_block)
        for idx_block in marked_block:
            grid_test[:, idx_block // self.grad_size[1], idx_block % self.grad_size[1]] = [0, 0, 0]

        while not Checking(grid_test[0, :, :]):
            grid_test = copy.deepcopy(grid_map)
            marked_block = random.sample(range(1, self.limit - 1), self.num_block)
            for idx_block in marked_block:
                grid_test[:, idx_block // self.grad_size[1], idx_block % self.grad_size[1]] = [0, 0, 0]
        grid_map = grid_test
        flag[marked_block] = 1

        return grid_map, flag

    def mark_in_out(self, grid_map):
        grid_map[:, 0, 0] = [20, 40, 222]
        grid_map[:, self.grad_size[1]-1, self.grad_size[1]-1] = [20, 223, 27]
        return grid_map


if __name__ == '__main__':
    PATH_DIR = 'Training_Maze'
    if not os.path.exists(os.path.join('..', PATH_DIR)):
        os.makedirs(os.path.join('..', PATH_DIR))
    else:
        shutil.rmtree(os.path.join('..', PATH_DIR))
        os.makedirs(os.path.join('..', PATH_DIR))

    maze_model = maze_build(NUM_BLOCKS, TOTAL_MAP, GRID_SIZE)

    for i in range(maze_model.total_map):
        graph, mark_flag = maze_model.maze_make()
        assert len(mark_flag) == maze_model.limit
        graph = maze_model.mark_in_out(graph)
        maze_model.maze_cache[i, ...] = graph
        maze_model.maze_flag[i, ...] = mark_flag

        maze_temp = Image.fromarray(np.uint8(graph.transpose(1, 2, 0)))     # C, H, W --> H, W, C
        # 不能保存为jpg格式，否则图像会压缩，导致图像灰度变化
        maze_temp.save(os.path.join("..", PATH_DIR, f'maze_{i}.png'))








