import colorsys
import random
import numpy as np
import math
import cv2
import os

import pathlib
LOCAL = pathlib.Path(__file__).parent
images = [ cv2.imread(LOCAL / "w12images" / path) for path in os.listdir(LOCAL / "w12images") ]

def showImage(title, i):
	max_h, h = len(i), 900
	max_w, w = len(i[0]), 1920

	mh = max_h / h
	mw = max_w / w

	m = max(mh, mw)
	if m != 1:
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

# Theta [0, 1] -> Color
def angleToColor(theta):
	return tuple(
		reversed(
			[ round(i*255) for i in colorsys.hsv_to_rgb(
				abs(theta)/(2*np.pi), 1, 1
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

	# vv THIS CAN WORK!! FIND THRESH VALUES SUCH THAT ALL TINY 1mm² SQUARES ARE SEPERATE,
	# THEN FIND ALL WHITE BLOBS (WHICH SHOULD INCLUDE SQUARES), FILTER FOR THE ACTUAL SQUARES
	# BY FINDING THEIR WIDTH AND HEIGHT AND ONLY ACCEPTING ONES WITH SIMILAR W AND H
	blurred = cv2.GaussianBlur(img, (5, 5), 0)
	# thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, THRESH_VALUES.get(i, 9), 2)
	thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
	thresh[thresh > 0] = 255
	# `thresh` has detected lines in (0,0,0) and everything else is (255,255,255) > invert for HoughLinesP
	thresh = cv2.bitwise_not(thresh)
	lines = cv2.HoughLinesP(thresh, rho=1, theta=np.pi/180, threshold=255, minLineLength=20, maxLineGap=1)

	out = thresh.copy()
	out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
	
	M = []

	for line in lines:
		if len(line[0]) == 4:
			x1, y1, x2, y2 = line[0]
			
			if x2 == x1:	m = np.inf
			else:			m = (y2-y1)/(x2-x1)
			M.append(float(m))
			
			# color = (255, 0, 0) if abs(m) > 10 else (0, 0, 255)
			color = angleToColor(math.atan(m))
			cv2.line(out, (x1, y1), (x2, y2), color, 2)

	M.sort()
	with open(LOCAL / "out.txt", "w+") as file: file.write("\n".join([ str(n) for n in M ]))

	# cv2.imshow(f"Threshold'd{i}", thresh)
	showImage(f"Found lines {i[0][:3]}", out)

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

def findDropWithBlobs(idx, i):
	# i = cv2.resize(i, None, fx=4, fy=4)
	# Filter out purple 
	hsv_image = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)

	# Dark Purple
	mask_1 = cv2.inRange(hsv_image, (270/360*180, 0, 0), (340/360*180, 255, 90))
	# Black
	mask_2 = cv2.inRange(hsv_image, (0, 0, 0), (255, 255, 20))
	blob_mask = cv2.bitwise_or(mask_1, mask_2)

	cv2.imwrite(LOCAL / "out0.png", blob_mask)
	showImage(f"Block Mask {idx}", blob_mask)

	_, _, stats, _ = cv2.connectedComponentsWithStats(blob_mask)
	blob = sorted(list(stats)[1:], key=lambda row: -row[4])[0]
	area = blob[4]
	print(f"Total area in pixels: {area}")

	return

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

# idx -> thresh value
THRESH_VALUES = {
	9:	31,	10:	31,	12:	31,	13:	31,	14:	31,	15:	31,	17:	31,	18:	31,	19:	31,
	16:	17
}

I = list(enumerate(images))[5:]
# random.shuffle(I)

for i, img in I:
	# findGridLines(img)
	findDropWithBlobs(i, img)
	print(i)
	break
	# continue

cv2.waitKey(0)

# cv2.imwrite(LOCAL / "out.png", lines_image)