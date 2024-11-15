import cv2
import numpy as np

import pathlib
LOCAL = pathlib.Path(__file__).parent
image = cv2.imread(LOCAL / "images" / "mathe_test.jpg")

# Crop to remove table
# image = image[37:2387, 56:7079]

# Filter out purple 
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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

lines_mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(lines_mask, 20, 120)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 1, np.array([]), 5, 1)

for line in lines:
	for x1, y1, x2, y2 in line:
		cv2.line(lines_mask, (x1, y1), (x2, y2), (255, 0, 255), 5)

cv2.imshow("Lines mask", lines_mask)

# cv2.imshow("Fancy output", cv2.resize(out, (960, 540)))
cv2.waitKey(0)