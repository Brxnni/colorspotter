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
	colors = [LIGHT_GREEN, YELLOW, RED]

	for i, row in enumerate(data):
		filename, area_cm2, error = row
		error_badness = min(badnesses[i], 3)
		error_color = colors[error_badness - 1]

		area_str = f"{area_cm2:.7f}... ≈ {LIGHT_BLUE}{area_cm2:.3f} cm^2{END}"
		line = f"{GRAY}|{END}"
		name = filename.split(".")[0]

		print(
			f"{BOLD}{name:>5}{END}",
			line,
			f"Area: {area_str:<17}",
			line,
			f"Error: {error_color}±{(error*100):.2f}%{END}"
		)

	avg_error = sum(errors)/len(errors)
	error_color = colors[np.digitize([avg_error], bin_edges)[0] - 1]
	print(
		f"{BOLD}Total{END}",
		line, " "*31, line,
		f"{BOLD}Average Error: {error_color}±{(avg_error*100):.2f}%{END}"
	)

def debug_histogram():
	import matplotlib.pyplot as plt

	# data_errors = [13.66, 9.38, 9.51, 8.92, 7.80, 7.93, 14.03, 6.66, 1.33, 2.52, 2.68, 2.90, 3.30, 3.14, 3.21, 3.62, 3.86, 3.83]
	data_errors = [1.33, 2.52, 2.68, 2.90, 3.14, 3.21, 3.30, 3.62, 3.83, 3.86, 6.66, 7.80, 7.93, 8.92, 9.38, 9.51, 13.66, 14.03]
	data_areas = [0.007429278094919611, 0.235600311361112, 0.5447433759004623, 0.7305454587516172, 0.9068696047017766, 0.9026456956792015, 0.1059966582211865, 0.796213106957912, 1.1860413379903059, 1.7807497844459985, 2.1204320248534776, 2.5771446046985824, 0.04830290387407343, 0.5088096921638484, 0.8076406268053976, 1.1933286139796424, 1.4741411627720886, 1.601330237028443]

	## Errors histogram
	# Categorized into good, medium and bad
	# hist, bins = np.histogram(data_errors, bins=3)
	# for i, v in enumerate(hist):
		# print(v, ["Good", "Mediocre", "Terrible"][i], "values")
	# Original data
	# hist, bins = np.histogram(data_errors, bins=range(len(data_errors)))

	# plt.hist(bins[:-1], bins, weights=hist, edgecolor="black")
	# plt.xlim(min(bins), max(bins))
	# plt.show()

	## Areas bar chart
	plt.bar(range(len(data_areas)), data_areas, edgecolor="black")
	plt.xlim(0, len(data_areas))
	plt.show()

if __name__ == "__main__": debug_histogram()
