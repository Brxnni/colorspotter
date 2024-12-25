import numpy as np

# Terminal colors
RED = "\033[91m"
YELLOW = "\033[93m"
LIGHT_GREEN = "\033[32m"
LIGHT_BLUE = "\033[36m"
GRAY = "\033[90m"
BOLD = "\033[1m"
END = "\033[0m"

def print_data(data):
	errors = [ row[2] for row in data ]
	_, bin_edges = np.histogram(errors, bins=3)
	badnesses = np.digitize(errors, bin_edges)

	for i, row in enumerate(data):
		filename, area_cm2, error = row
		error_badness = min(badnesses[i], 3)
		error_color = [LIGHT_GREEN, YELLOW, RED][error_badness - 1]

		area_str = f"{area_cm2:.7f}... ≈ {LIGHT_BLUE}{area_cm2:.3f} cm^2{END}"
		print(
			BOLD + filename.split(".")[0] + END,
			f"{GRAY}|{END}",
			f"Area: {area_str:<17}",
			f"{GRAY}|{END}",
			f"Error: {error_color}±{(error*100):.2f}%{END}"
		)

	avg_error = sum(errors)/len(errors)
	print(f"Total average error: {BOLD}{YELLOW}{(avg_error*100):.2f}%{END}")

def debug_histogram():
	import matplotlib.pyplot as plt

	# data = [13.66, 9.38, 9.51, 8.92, 7.80, 7.93, 14.03, 6.66, 1.33, 2.52, 2.68, 2.90, 3.30, 3.14, 3.21, 3.62, 3.86, 3.83]
	data = [1.33, 2.52, 2.68, 2.90, 3.14, 3.21, 3.30, 3.62, 3.83, 3.86, 6.66, 7.80, 7.93, 8.92, 9.38, 9.51, 13.66, 14.03]

	# Categorized into good, medium and bad
	hist, bins = np.histogram(data, bins=3)
	for i, v in enumerate(hist):
		print(v, ["Good", "Mediocre", "Terrible"][i], "values")

	# Original data
	# hist, bins = np.histogram(data, bins=range(len(data)))

	plt.hist(bins[:-1], bins, weights=hist, edgecolor="black")
	plt.xlim(min(bins), max(bins))
	plt.show()

if __name__ == "__main__": debug_histogram()
