import colorsys
import random
import numpy as np
import math
# import json
import cv2
import os

import pathlib
LOCAL = pathlib.Path(__file__).parent
IMAGES = LOCAL / "w12_images"
images = [ cv2.imread(IMAGES / path) for path in os.listdir(IMAGES) ]

# Common use colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)

# Scales image before calling `imshow` down so it fits on my monitor (only 2k smh)
def showImage(title, i, scale=True):
	max_h, h = 950, len(i),
	max_w, w = 1900, len(i[0]),

	mh = h / max_h
	mw = w / max_w

	m = max(mh, mw)
	if m != 1 and scale:
		i = cv2.resize(i, None, fx=1/m, fy=1/m)
	
	cv2.imshow(title, i)

def randomColor():
	return tuple(
		# RGB -> BGR
		reversed(
			# RGB(normalized) -> RGB(0-255,0-255,0-255)
			[ round(i*255) for i in colorsys.hsv_to_rgb(
				# HSV(normalized) -> RGB(normalized)
				random.random(), 1, 1)
			]
		)
	)

# Theta [-π, π] -> Bright Color
def angleToColor(θ):
	# -> [0, 2π]
	θ += np.pi
	return tuple(
		reversed(
			[ round(i*255) for i in colorsys.hsv_to_rgb(
				θ/(2*np.pi), 1, 1
			)]
		)
	)

def isSquare(contour):
	area = cv2.contourArea(contour)
	x, y, width, height = cv2.boundingRect(contour)
	aspect_ratio = float(width) / height

	hull = cv2.convexHull(contour)
	hull_area = cv2.contourArea(hull)
	if hull_area == 0: return False
	solidity = float(area) / hull_area

	print(solidity)
	# return solidity > 0.95
	return abs(aspect_ratio - 1) < 0.1

def findGridLines(idx, i):
	img = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
	# Find red-ish pixels (lines)
	lower = np.array([10/360*180, 120, 70])
	upper = np.array([50/360*180, 245, 200])
	# ^ these are magic values, change these and nothing works :D
	mask = cv2.inRange(img, lower, upper)

	# Filter for only red-ish pixels, make everything else black
	img = cv2.bitwise_and(img, img, mask=mask)

	# HSV -> Grayscale to find lines
	bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
	gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
	lines = cv2.HoughLinesP(gray, rho=1, theta=np.pi/180, threshold=255, minLineLength=80, maxLineGap=7)

	lines = [{
		# Point 1 & 2
		"pt1": (x1, y1),
		"pt2": (x2, y2),
		"x1": x1, "y1": y1,
		"x2": x2, "y2": y2,
		# Angle & Slope
		"angle": abs(math.atan2(-(y2-y1), x2-x1))
	} for line in lines for (x1, y1, x2, y2) in line ]

	# Remove lines with Dx=0
	# lines = [ line for line in lines if line[0][0] != line[0][1] ]

	# with open(LOCAL / "lines_slopes.json", "r") as file:
	# 	data = json.loads(file.read())
	# with open(LOCAL / "lines_slopes.json", "w") as file:
	# 	data.append([ line["angle"] for line in lines ])
	# 	file.write(json.dumps(data))

	hist, bin_edges = np.histogram([ line["angle"] for line in lines ], bins = 10)

	# Find two most common angles (these correspond to the vertical and horizontal lines)
	bin_idx1, bin_idx2 = np.argpartition(hist, -2)[-2:]
	θ_range1 = (bin_edges[bin_idx1], bin_edges[bin_idx1+1])
	θ_range2 = (bin_edges[bin_idx2], bin_edges[bin_idx2+1])

	# Find lines that have those angles (i.e. find vertical and horizontal lines)
	lines1 = [ line for line in lines if θ_range1[0] <= line["angle"] <= θ_range1[1] ]
	lines2 = [ line for line in lines if θ_range2[0] <= line["angle"] <= θ_range2[1] ]

	# Debug stuff
	out = img.copy()
	out = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)

	for line in lines:
		cv2.line(out, line["pt1"], line["pt2"], randomColor(), 2)
	# for line in lines1:
	# 	cv2.line(out, line["pt1"], line["pt2"], MAGENTA, 2)
	# for line in lines2:
		# cv2.line(out, line["pt1"], line["pt2"], YELLOW, 2)

	# d = abs(t2-t1)/sqrt(1+m**2)
	# Find average y-intercept difference

	# Filter out lines with Dx=0 (i'm too lazy to calculate x-intercepts)
	y_intercepts1 = [ line for line in lines1 if line["x1"] != line["x2"] ]
	y_intercepts2 = [ line for line in lines2 if line["x1"] != line["x2"] ]
	
	# y_intercepts = max(y_intercepts1, y_intercepts2, key=len)
	y_intercepts = y_intercepts1

	# t = y0-mx0
	y_intercepts = sorted([ line["y1"] - (line["y2"]-line["y1"])/(line["x2"]-line["x1"])*line["x1"] for line in y_intercepts ])
	
	for y in y_intercepts:
		cv2.line(out, (0, round(y)), (30, round(y)), RED, 2)
	
	distances = np.diff(y_intercepts)
	# distances = [ d for d in distances if d > 10 ]
	for d in distances: print(d)
	# print(y_intercepts)
	print(">>>", np.mean(distances))

	showImage(f"Detected Lines {idx}", out)

def findDropWithComps(idx, i):
	# i = cv2.resize(i, None, fx=4, fy=4)
	# Filter out purple 
	hsv_image = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)

	# Dark Purple
	mask_1 = cv2.inRange(hsv_image, (270/360*180, 0, 0), (340/360*180, 255, 90))
	# Black (Dark Gr[ae]y)
	mask_2 = cv2.inRange(hsv_image, (0, 0, 0), (255, 255, 20))
	blob_mask = cv2.bitwise_or(mask_1, mask_2)

	showImage(f"Blob Mask {idx}", blob_mask)
	cv2.imwrite(LOCAL / "blobmask.png", blob_mask)

	_, _, stats, _ = cv2.connectedComponentsWithStats(blob_mask)
	blob = sorted(list(stats)[1:], key=lambda row: -row[4])[0]
	area = blob[4]
	print(f"Total area in pixels: {area}")

I = list(enumerate(images))[0:]
random.shuffle(I)

# with open(LOCAL / "lines_slopes.json", "w") as file:
	# file.write("[]")

for i, img in I:
	findGridLines(i, img)
	# findDropWithComps(i, img)
	# print(i)
	break
	# continue

cv2.waitKey(0)