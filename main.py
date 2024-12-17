from shapely.geometry import LineString
import colorsys
import random
import numpy as np
import math
import cv2
import os

import pathlib
LOCAL = pathlib.Path(__file__).parent
IMAGES = LOCAL / "w12_images"
filenames = [ path for path in os.listdir(IMAGES) ]
images = [ cv2.imread(IMAGES / filename) for filename in filenames ]

# Commonly used colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)

# Terminal colors
T_YELLOW = "\033[93m"
T_RED = "\033[91m"
T_LIGHT_BLUE = "\033[36m"
T_GRAY = "\033[90m"
T_BOLD = "\033[1m"
END = "\033[0m"

def find_mode(nums, epsilon):
	nums.sort()
	prev = nums[-1]
	mode = 0
	maxcount = 0
	current = 0

	for n in nums:
		if abs(n - prev) <= epsilon:
			current += 1
			if current > maxcount:
				maxcount = current
				mode = n
		else:
			current = 1
		prev = n
	
	return mode

# Scales image before calling `imshow` down so it fits on my monitor (only 2k smh)
def showImage(title, img, scale=True):
	max_h, max_w = 950, 1900
	h, w = img.shape[:2]

	mh = h / max_h
	mw = w / max_w

	m = max(mh, mw)
	if m != 1 and scale:
		img = cv2.resize(img, None, fx=1/m, fy=1/m)
	
	cv2.imshow(title, img)

def randomColor():
	return tuple(reversed([ round(i*255) for i in colorsys.hsv_to_rgb(
		random.random(), 1, 1
	)]))

# Angle [-π, π] -> Bright Color
def angleToColor(θ):
	return tuple(reversed([ round(i*255) for i in colorsys.hsv_to_rgb(
		(θ+np.pi)/(2*np.pi), 1, 1
	)]))

# Position [0, 1] -> Bright color
def coordToColor(normalized):
	return tuple(reversed([ round(i*255) for i in colorsys.hsv_to_rgb(
		(normalized * 3) % 1, 1, 1
	)]))

def intersectionDistance(lines1, lines2):
	lines1 = [ LineString([line["pt1"], line["pt2"]]) for line in lines1 ]
	lines2 = [ LineString([line["pt1"], line["pt2"]]) for line in lines2 ]

	points = []
	for line1 in lines1:
		for line2 in lines2:

			if line1.intersects(line2):
				intersection = line1.intersection(line2)
				points.append((intersection.x, intersection.y))

	# This algorithm is about as stable as a house of cards
	neighbour_distances = []
	for point in points:
		distances = []
		for other_point in points:
			distance = math.hypot(abs(point[0]-other_point[0]), abs(point[1]-other_point[1]))
			# Magic number (?), this one is fairly straight-forward though
			if distance <= 20: continue
			distances.append(distance)

		distances.sort()
		shortest = distances[:4]
		avg_neighbour_distance = sum(shortest) / 4
		neighbour_distances.append(avg_neighbour_distance)

	neighbour_distances.sort()
	mode = find_mode(neighbour_distances, 0.2)

	return points, mode

def findGridSize(filename, i, debug=False):
	img = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
	# Find red-ish pixels (lines)
	# these are magic values, change these and nothing works :D
	lower = np.array([5/360*180, 100, 70])
	upper = np.array([50/360*180, 245, 200])
	mask = cv2.inRange(img, lower, upper)

	# Filter for only red-ish pixels, make everything else black
	filtered = cv2.bitwise_and(img, img, mask=mask)

	# HSV -> Grayscale
	bgr = cv2.cvtColor(filtered, cv2.COLOR_HSV2BGR)
	gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
	# find lines
	lines = cv2.HoughLinesP(gray, rho=1, theta=np.pi/180, threshold=255, minLineLength=80, maxLineGap=7)

	lines = [{
		# Point 1 & 2
		"pt1": (x1, y1),
		"pt2": (x2, y2),
		"x1": x1, "y1": y1,
		"x2": x2, "y2": y2,
		# Angle & Slope
		# (-y2-y1) because in opencv, the origin is at the top left
		# and atan2 thinks it should be at the bottom
		"angle": abs(math.atan2(-(y2-y1), x2-x1))
	} for line in lines for (x1, y1, x2, y2) in line ]

	# Magic number: 25 bins seems to work the best for these images
	hist, bin_edges = np.histogram([ line["angle"] for line in lines ], bins=25)

	# Find two most common angles (these correspond to the vertical and horizontal lines)
	bin_idx1, bin_idx2 = np.argpartition(hist, -2)[-2:]
	θ_range1 = (bin_edges[bin_idx1], bin_edges[bin_idx1+1])
	θ_range2 = (bin_edges[bin_idx2], bin_edges[bin_idx2+1])

	# Find lines that have those angles (i.e. find vertical and horizontal lines)
	lines1 = [ line for line in lines if θ_range1[0] <= line["angle"] <= θ_range1[1] ]
	lines2 = [ line for line in lines if θ_range2[0] <= line["angle"] <= θ_range2[1] ]

	# Find most common distance between intersection points of vertical and horizontal lines
	# => thats 1cm
	points, cm_distance = intersectionDistance(lines1, lines2)

	# SPECIAL CASE: Only in these two images (everywhere else it works perfectly),
	# the program returns the distance between the 5mm lines (because it detected them too),
	# so the distance needs to be doubled
	# This makes me a bad opencv developer
	if filename[:2] in ["B1", "B2"]:
		cm_distance *= 2

	if debug:
		out = img.copy()
		out = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)

		for line in [*lines1, *lines2]:
			cv2.line(out, line["pt1"], line["pt2"], BLACK, 5)

		for point in points:
			x, y = point
			x, y = int(x), int(y)
			cv2.circle(out, (x, y), 8, WHITE, 3, cv2.LINE_AA)
			cv2.line(out, (x, y), (x, y + int(cm_distance)), coordToColor(y/img.shape[0]), 6)

		# Display length of line in pixels next to one line (the most top-left one that has enough space above)
		x, y = sorted([p for p in points if p[1] > 120])[0]
		pos_x = int(x)
		pos_y = int(y - 40)
		text = f"d = {round(cm_distance, 3):,}px".replace(",", " ").replace(".", ",")

		cv2.putText(out, text, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, BLACK, 10, cv2.LINE_AA)
		cv2.putText(out, text, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, WHITE, 4, cv2.LINE_AA)

		showImage(f"{filename} :: Detected Lines", out)
		new_filename = filename.split(".")[0] + "_lines.png"
		cv2.imwrite(LOCAL / "w12_out" / new_filename, out)
	
	return cm_distance

def findDropArea(filename, img, debug=False):
	# Filter out specific shade of purple that belongs to the KMnO4 crystals 
	hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# Dark Purple
	mask_1 = cv2.inRange(hsv_image, (270/360*180, 0, 0), (340/360*180, 255, 90))
	# Black and Dark Gray
	mask_2 = cv2.inRange(hsv_image, (0, 0, 0), (180, 255, 47)) # <- last V value is this and not 30 because of A6
	blob_mask = cv2.bitwise_or(mask_1, mask_2)

	# Get rid of tiny sharp edges where the blob touches lines
	# A1 is so fucking tiny that smoothing would get rid of the blob completely
	if not filename.startswith("A1"):
		blob_mask = cv2.morphologyEx(blob_mask, cv2.MORPH_OPEN, np.ones((3, 3)))

	# Find continous blobs of color
	contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	blobs = [ cv2.contourArea(contour) for contour in contours ]
	# A6 is weird because the red line is visible through the purple blob,
	# so we need to get the convex hull to account for the weird shape (see A6_blob.png)
	if filename[:2] == "A6":
		blobs = [ cv2.contourArea(cv2.convexHull(contour)) for contour in contours ]

	# Biggest blob is the one we want
	blobs.sort()
	area = blobs[-1]

	if debug:
		# Display contours with colorful lines
		for c in contours:
			cv2.drawContours(img, [c], 0, randomColor(), 2)

		# Display area of blob in pixels above it
		x, y, _, _ = cv2.boundingRect(sorted(contours, key=cv2.contourArea)[-1])
		pos_x = x
		pos_y = y - 40
		text = f"A = {int(area):,}px".replace(",", " ")

		# Hack to draw text with shadow
		cv2.putText(img, text, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, BLACK, 10, cv2.LINE_AA)
		cv2.putText(img, text, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, WHITE, 4, cv2.LINE_AA)

		showImage(f"{filename} :: Detected Blob", img)
		new_filename = filename.split(".")[0] + "_blob.png"
		cv2.imwrite(LOCAL / "w12_out" / new_filename, img)
	
	return area

errors = []
DEBUG = False

for i, (filename, img) in enumerate(zip(filenames, images)):
	distance_px = findGridSize(filename, img, DEBUG)
	area_px = findDropArea(filename, img, DEBUG)
	area_cm2 = area_px/(distance_px**2)

	# ±3.5 pixels for line finding due to lines themselves being around 7 pixels wide
	distance_error_abs = 3.5
	distance_error_rel = distance_error_abs / distance_px
	# Total error: Area error (≈0) + 2 * Distance Error
	total_error_rel = 2*distance_error_rel
	errors.append(total_error_rel)

	area_str = f"{area_cm2:.7f}... ≈ {T_LIGHT_BLUE}{area_cm2:.3f} cm^2{END}"
	print(
		T_BOLD + filename.split(".")[0] + END,
		f"{T_GRAY}|{END}",
		f"Area: {area_str:<17}",
		f"{T_GRAY}|{END}",
		f"Error: {T_YELLOW}±{(total_error_rel*100):.2f}%{END}"
	)

print(f"Total average error: {T_BOLD}{T_YELLOW}{(sum(errors)/len(errors)*100):.2f}%{END}")

if DEBUG: cv2.waitKey(0)
