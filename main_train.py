# -*- coding: utf-8 -*-
# @Time    : 2022/1/10 15:38
# @Author  : Scotty
# @FileName: main_train.py
# @Software: PyCharm
import sys
import os
import numpy as np
import pandas as pd
from PIL import Image
from utils_tools.DDQN import DDQN
from utils_tools.initialize import maze_model

class Env:
    def __init__(self):
        self.