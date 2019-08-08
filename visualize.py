import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
import sys

label = ['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'elapsed_time']

ax = [
  plt.subplot2grid((3, 3), (0, 0)),
  plt.subplot2grid((3, 3), (2, 0), colspan=3),
  plt.subplot2grid((3, 3), (0, 1)),
  plt.subplot2grid((3, 3), (0, 2)),
  plt.subplot2grid((3, 3), (1, 0)),
  plt.subplot2grid((3, 3), (1, 1)),
  plt.subplot2grid((3, 3), (1, 2)),
]

plt.tight_layout()

with open('storage/logs.json') as json_file:
  data = json.load(json_file)
  data = np.array(data).T
  for t in range(len(label[0])):
    for i in range(len(label)):
      x0 = np.linspace(0, 1, len(data[0]))
      ax[i].set_title(label[i])
      ax[i].plot(x0, data[i], 'r-', linewidth = 1)

plt.show()