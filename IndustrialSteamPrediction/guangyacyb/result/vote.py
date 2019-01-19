#!/usr/bin/python3
# 投票，只是简单的求了平均
import numpy as np

filename = ["carttree.txt","ringe.txt","xgb.txt"]

file1 = open(filename[0], "r")
file2 = open(filename[1], "r")
file3 = open(filename[2], "r")

lines1 = file1.readlines()
lines2 = file2.readlines()
lines3 = file3.readlines()

if len(lines1) != len(lines2) or len(lines2) != len(lines3):
    exit(-1)

result = []
for i in range(len(lines1)):
    res = (float(lines1[i]) + float(lines2[i]) + float(lines3[i]) ) / 3
    result.append(res)
print (len(result))
np.savetxt('./final.txt', result)
