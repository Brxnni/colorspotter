import colorsys
import random
import numpy as np
import math
import cv2
import os

import pathlib
LOCAL = pathlib.Path(__file__).parent
images = [ cv2.imread(LOCAL / "w12images" / path) for path in os.listdir(LOCAL / "w12images") ]

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
	a = sorted([ cv2.contourArea(c) for c in contours ])
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
	# lines = cv2.HoughLines(thresh, rho=1, theta=np.pi/180, threshold=156, lines=None, srn=0, stn=0)
	lines = cv2.HoughLinesP(thresh, rho=1, theta=np.pi/180, threshold=255, minLineLength=20, maxLineGap=1)

	# print(i, "|", len(lines))

	out = thresh.copy()
	out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
	
	for line in lines:
		# HoughLines
		if len(line[0]) == 2:
			rho, theta = line[0]
			a = math.cos(theta)
			b = math.sin(theta)
			x0 = a * rho
			y0 = b * rho
			pt1 = (int(x0 + 30*(-b)), int(y0 + 30*(a)))
			pt2 = (int(x0 - 30*(-b)), int(y0 - 30*(a)))

			cv2.line(out, pt1, pt2, randomColor(), 1)
		# HoughLinesP
		elif len(line[0]) == 4:
			for x1, y1, x2, y2 in line:
				cv2.line(out, (x1, y1), (x2, y2), randomColor(), 2)

	# cv2.imshow(f"Threshold'd{i}", thresh)
	cv2.imshow(f"Found lines {i}", out)

# Filter out purple 
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# blob_mask = cv2.inRange(hsv_image, (135, 0, 0), (160, 255, 255))
# contours = cv2.findContours(
# 	cv2.threshold(blob_mask, 128, 255, cv2.THRESH_BINARY)[1],
# 	cv2.RETR_EXTERNAL,
# 	cv2.CHAIN_APPROX_SIMPLE
# )
# contours = contours[0] if len(contours) == 2 else contours[1]
# contours = [ c for c in contours if cv2.contourArea(c) > 500 ]
# contours = sorted(contours, key = lambda c: -cv2.contourArea(c))

# Visualize found blobs
# out = cv2.bitwise_and(image, image, mask=blob_mask)
# for c in contours:
# 	cv2.drawContours(out, [c], 0, (0, 0, 255), 2)

# Calculate area of blobs
# total_area = sum([ cv2.contourArea(c) for c in contours ])
# print(f"Total area in pixels: {total_area}")

# idx -> thresh value
THRESH_VALUES = {
	9:	31,	10:	31,	12:	31,	13:	31,	14:	31,	15:	31,	17:	31,	18:	31,	19:	31,
	16:	17
}

# random.shuffle(images)

for i, img in enumerate(images):
	findGridLines(img)	
	# continue
	# break
	
cv2.waitKey(0)

# cv2.imwrite(LOCAL / "out.png", lines_image)