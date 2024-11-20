# This file is only used for debugging purposes
import matplotlib.pyplot as plt
import numpy as np
import json

import pathlib
LOCAL = pathlib.Path(__file__).parent

with open(LOCAL / "lines_slopes.json", "r") as file:
	slope_data = json.loads(file.read())
with open(LOCAL / "lines_intercepts.json", "r") as file:
	intercept_data = json.loads(file.read())

# for slopes in slope_data:
# 	hist, bins = np.histogram(slopes, bins=10)
# 	plt.hist(slopes, len(bins), histtype="step", stacked=True, fill=False)

for slopes in intercept_data:
	hist, bins = np.histogram(slopes, bins=300)
	plt.hist(slopes, len(bins), histtype="step", stacked=True, fill=False)

plt.show()