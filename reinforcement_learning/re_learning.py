# -*- coding: utf-8 -*-

# @Time    : 2019-04-16 15:36
# @Author  : jian
# @File    : re_learning.py
import numpy as np
from reinforcement_learning.gridworld import GridworldEnv

env = GridworldEnv()

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    def ones_step_lookahead(state, v):
        A = np.zeros(env.nA)
        for a in range(4):
            for prob, next_state, reward, done in env.p[state][a]:
                A[a] += prob*(reward + discount_factor*v[next_state])
        return A
    v = np.zeros(env.nS)

