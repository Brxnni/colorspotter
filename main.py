import colorsys
import random
import numpy as np
import math
import cv2
import os

import pathlib
LOCAL = pathlib.Path(__file__).parent
images = [ cv2.imread(LOCAL / "w12images" / path) for path in os.listdir(LOCAL / "w12images") ]

def showImage(title, i, scale=True):
	max_h, h = len(i), 900
	max_w, w = len(i[0]), 1920

	mh = max_h / h
	mw = max_w / w

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

def findSquareContours(i):
	img = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

	blurred = cv2.GaussianBlur(img, (5, 5), 0)
	thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, THRESH_VALUES.get(i, 9), 2)
	thresh[thresh > 0] = 255

	# Find lines in thresh
	contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	contours = [ c for c in contours if cv2.contourArea(c) > 10 ]
	contours = [ c for c in contours if isSquare(c) ]
	print(i, "|", len(contours))
	
	thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
	for idx, contour in enumerate(contours):
		cv2.drawContours(image=thresh, contours=contours, contourIdx=idx, color=randomColor(), thickness=1)

	cv2.imshow(f"Detected Edges{i}", thresh)

def findGridLines(i):
	img = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

	blurred = cv2.GaussianBlur(img, (11, 11), 0)
	# blurred = cv2.dilate(blurred, np.ones((5, 5)))
	# blurred = img.copy()
	# thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, THRESH_VALUES.get(i, 9), 2)
	thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
	thresh[thresh > 0] = 255
	# `thresh` has detected lines in (0,0,0) and everything else is (255,255,255) > invert for HoughLinesP
	thresh = cv2.bitwise_not(thresh)
	lines = cv2.HoughLinesP(thresh, rho=1, theta=np.pi/180, threshold=255, minLineLength=20, maxLineGap=1)

	out = thresh.copy()
	out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

	lines_objs = []
	for line in lines:
		for (x1, y1, x2, y2) in line:
			# In OpenCV, (0, 0) is in the top left, but we calculate the slope as if it was in the bottom left
			# -> dx = -(x2-x1) instead of x2-x1
			θ = abs(math.atan2(y2-y1, -(x2-x1)))
			
			lines_objs.append({
				"p1": (x1, y1),
				"p2": (x2, y2),
				"θ": θ
			})

	# print([ line["θ"] for line in lineslines_objs ])

	hist, bin_edges = np.histogram([ obj["θ"] for obj in lines_objs ], bins = 10)

	# Find two most common angles (these correspond to the vertical and horizontal lines)
	bin_idx1, bin_idx2 = np.argpartition(hist, -2)[-2:]
	θ_range1 = (bin_edges[bin_idx1], bin_edges[bin_idx1+1])
	θ_range2 = (bin_edges[bin_idx2], bin_edges[bin_idx2+1])

	# Find lines that have those angles (i.e. find vertical and horizontal lines)
	lines1 = [ line for line in lines_objs if θ_range1[0] <= line["θ"] <= θ_range1[1] ]
	lines2 = [ line for line in lines_objs if θ_range2[0] <= line["θ"] <= θ_range2[1] ]

	# Debug drawing
	# Green/Pink = Horizontal/Vertical, Blue = Neither, irrelevant
	for line in lines_objs:
		# if line not in lines1 and lines2:
		cv2.line(out, line["p1"], line["p2"], (255, 0, 0), 2)
	for line in lines1:
		cv2.line(out, line["p1"], line["p2"], (255, 0, 255), 2)
	for line in lines2:
		cv2.line(out, line["p1"], line["p2"], (0, 255, 0), 2)

	# Find t-values of all lines so that we can find the average distance between them
	# (i.e. the cell size of the graph paper)
	t_values1 = [  ]

	# !!; The only things left to do are: Finish the calculation of the t-values of all the lines (see purpose above),
	# do the final calculation for the area of the blob (very simple)
	# But most importantly, improve the line finding process, either by making the `thresh` image better or by choosing bettrr params for HoughLinesP

	print(hist, bin_edges)
	print(θ_range1, θ_range2)

	# cv2.imshow(f"Threshold'd{i}", thresh)
	showImage(f"Found lines {i[0][:3]}", out, True)
	cv2.imwrite(LOCAL / "gridlines.png", out)

def isDropShaped(c):
	area = cv2.contourArea(c)
	hull = cv2.convexHull(c)

	hull_area = cv2.contourArea(hull)
	if hull_area == 0: return False
	solidity = float(area) / hull_area
	return solidity >= 0.75

def findDropWithContours(idx, i):
	# i = cv2.resize(i, None, fx=4, fy=4)
	# Blur
	# Filter out purple 
	hsv_image = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)

	# Dark Purple
	mask_1 = cv2.inRange(hsv_image, (270/360*180, 0, 0), (340/360*180, 255, 90))
	# Black
	mask_2 = cv2.inRange(hsv_image, (0, 0, 0), (255, 255, 20))
	blob_mask = cv2.bitwise_or(mask_1, mask_2)
	# blob_mask = cv2.dilate(blob_mask, np.ones((4, 4)))

	cv2.imwrite(LOCAL / "out0.png", blob_mask)
	showImage(f"Block Mask {idx}", blob_mask)

	contours = cv2.findContours(
		cv2.threshold(blob_mask, 128, 255, cv2.THRESH_BINARY)[1],
		cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE
	)
	contours = contours[0] if len(contours) == 2 else contours[1]

	print(len(contours))
	contours = [ c for c in contours if isDropShaped(c) ]
	print(len(contours))
	contours = [ c for c in contours if cv2.contourArea(c) > 2 ]
	print(len(contours))
	contours = sorted(contours, key = lambda c: -cv2.contourArea(c))
	print([ cv2.contourArea(a) for a in contours ])

	# Visualize found blobs
	out = i.copy()
	# out = cv2.bitwise_and(i, i, mask=blob_mask)
	for idx, c in enumerate(contours):
		color = (255, 0, 0) if idx == 0 else (0, 0, 255)
		cv2.drawContours(out, contours, idx, color, 5)
	showImage(f"Found purple drop {idx}", out)
	cv2.imwrite(LOCAL / "out1.png", out)

	total_area = 0
	if len(contours) > 0:
		contours = contours[0]
		# Calculate area of blobs
		total_area = sum([ cv2.contourArea(c) for c in contours ])
	print(f"Total area in pixels: {total_area}")

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
# random.shuffle(I)

for i, img in I:
	findGridLines(img)
	# findDropWithComps(i, img)
	print(i)
	# break
	# continue

cv2.waitKey(0)