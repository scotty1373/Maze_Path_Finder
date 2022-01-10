# -*- coding: utf-8 -*-
# @Time    : 2022/1/8 17:34
# @Author  : Scotty
# @FileName: DDQN.py
# @Software: PyCharm
import torch
import sys
import os
import numpy as np
from collections import deque
import random
import copy
from model import Model
from torch.autograd import Variable


class DDQN:
    def __init__(self, state_size_, action_size_, device_):
        self.t = 0
        self.max_Q = 0
        self.train_loss = 0
        self.train = True
        self.train_from_ckpt = False

        # Get size of state and action
        self.state_size = state_size_
        self.action_size = action_size_
        self.device = device_

        # Epsilon greddy  based alg parameters initialize
        if self.train and not self.train_from_ckpt:
            self.epsilon = 1.0
            self.initial_epsilon = 1.0
        else:
            self.epsilon = 0
            self.initial_epsilon = 0
        self.epsilon_min = 0.01

        # trainner initialize
        self.batch_size = 64
        self.train_start = 100
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        self.train_from_checkpoint_start = 3000
        self.explore = 4000

        # Create replay memory using deque
        self.memory = deque(maxlen=32000)

        # Create main model and target model
        self.model = Model(input_shape=self.state_size, out_dim=2).to(self.device)
        self.target_model = Model(input_shape=self.state_size, out_dim=2).to(self.device)

        # Deepcopy the model to target model
        self.update_target_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss = torch.nn.MSELoss()

    def update_target_model(self):
        # 解决state_dict浅拷贝问题
        weight_model = copy.deepcopy(self.model.state_dict())
        self.target_model.load_state_dict(weight_model)

    # Get action from model using epsilon-greedy policy
    def get_action(self, Input):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, 1)
        else:
            q_value = self.model(Input[0], Input[1])
            return torch.argmax(q_value[0]).item()

    def replay_memory(self, state, v_ego, action, reward, next_state, nextV_ego, done):
        self.memory.append((state, action, reward, next_state, done, self.t))
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.initial_epsilon - self.epsilon_min) / self.explore

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        state_t, v_ego_t, action_t, reward_t, state_t1, v_ego_t1, terminal, step = zip(*minibatch)
        state_t = Variable(torch.Tensor(state_t).squeeze().to(self.device))
        state_t1 = Variable(torch.Tensor(state_t1).squeeze().to(self.device))

        self.optimizer.zero_grad()

        targets = self.model(state_t)
        self.max_Q = torch.max(targets[0]).item()
        target_val = self.model(state_t1)
        target_val_ = self.target_model(state_t1)
        for i in range(batch_size):
            if terminal[i] == 1:
                targets[i][action_t[i]] = reward_t[i]
            else:
                a = torch.argmax(target_val[i])
                targets[i][action_t[i]] = reward_t[i] + self.discount_factor * (target_val_[i][a])
        logits = self.model(state_t, v_ego_t)
        loss = self.loss(logits, targets)
        loss.backward()
        self.optimizer.step()
        self.train_loss = loss.item()

    def load_model(self, name):
        checkpoints = torch.load(name, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(checkpoints['model'])
        self.optimizer.load_state_dict(checkpoints['optimizer'])

    # Save the model which is under training
    def save_model(self, name):
        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, name)

