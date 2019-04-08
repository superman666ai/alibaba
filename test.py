# -*- coding: utf-8 -*-

# @Time    : 2019-04-08 16:03
# @Author  : jian
# @File    : test.py

import tensorflow as tf
import numpy as np

labels = np.array([[1, 1, 1, 0],
                   [1, 1, 1, 0],
                   [1, 1, 1, 0],
                   [1, 1, 1, 0]], dtype=np.uint8)

predictions = np.array([[1, 0, 0, 0],
                        [1, 1, 0, 0],
                        [1, 1, 1, 0],
                        [0, 1, 1, 1]], dtype=np.uint8)

n_batches = len(labels)

# First,calculate precision over entire set of batches
# using formula mentioned above
# pred_p = (predictions > 0).sum()
#
# print(pred_p)
#
# true_p = (labels * predictions > 0).sum()
# print(true_p)
#
# precision = true_p / pred_p
# print(precision)
# # print("Precision :%1.4f" % (precision))



def reset_running_variables():
    """ Resets the previous values of running variables to zero """
    global N_TRUE_P, N_PRED_P
    N_TRUE_P = 0
    N_PRED_P = 0


def update_running_variables(labs, preds):
    global N_TRUE_P, N_PRED_P
    N_TRUE_P += ((labs * preds) > 0).sum()
    N_PRED_P += (preds > 0).sum()


def calculate_precision():
    global N_TRUE_P, N_PRED_P
    return float(N_TRUE_P) / N_PRED_P



# reset_running_variables()
# for i in range(n_batches):
#     update_running_variables(labs=labels[i], preds=predictions[i])
#
# precision = calculate_precision()
# print("[NP] SCORE: %1.4f" %precision)
#

# Batch precision
for i in range(n_batches):
    reset_running_variables()
    update_running_variables(labs=labels[i], preds=predictions[i])
    prec = calculate_precision()
    print("- [NP] batch %d score: %1.4f" %(i, prec))