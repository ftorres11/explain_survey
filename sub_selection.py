# -*- coding: utf-8 -*-
import random
import numpy as np

import os
import pdb

osp = os.path
osj = osp.join
np.random.seed(68)


permutation = np.random.permutation(1000)
chosen = permutation[0:200]
classes_ids = {}

# Gathering 1k classes
for elem in np.arange(0, 1000):
    classes_ids[str(elem)] = []

# Reading the file
with open(osj('..', '50k_sorted.csv'), 'r') as data:
    lines = data.readlines()

# Dictionary with each class and its corresponding files
for line in lines:
    name, cat = line.strip().split(',')
    classes_ids[cat].append(line)

# Selecting 5 instances per class, out of the 200 randomly chosen classes
with open('200classes_5inst.csv', 'w') as data:
    for class_wnid in chosen:
        for ix in np.arange(0,5):
            line = classes_ids[str(class_wnid)][ix]
            data.write('{}'.format(line))
