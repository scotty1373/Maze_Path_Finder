# -*- coding: utf-8 -*-
# @Time    : 2022/1/8 17:35
# @Author  : Scotty
# @FileName: maze_build.py
# @Software: PyCharm
import numpy as np


# 判断当前位置是否可达且为第一次到达，像素为白色为可达，可达返回True，否则返回False
def Valid(i, limit, flag, value):
    if 0 <= i <= limit - 1 and value[i] == 255 and flag[i] == 0:
        return True
    return False


def Checking(graph):
    stack = []
    x, y = graph.shape
    # flatten 图像像素
    graph1 = np.reshape(graph, -1)
    limit = len(graph1)
    # DFS算法走过当前位置次数记录
    flag = np.zeros([len(graph1)])

    # 左上角起始点指定白色通路， 标志位标定1
    stack.append(0)
    flag[0] = 1
    path_check = []

    while len(stack) > 0 and not flag[-1] == 1:
        element = stack.pop()
        # 控制右侧边界，防止element+1 超过右侧界限导致重新记录下一行
        if Valid(element + 1, limit, flag, graph1) and not (element + 1) % x == 0:
            stack.append(element + 1)
            flag[element + 1] = 1

        if Valid(element + x, limit, flag, graph1):
            stack.append(element + x)
            flag[element + x] = 1

    if flag[-1] == 1:
        return True
    return False


