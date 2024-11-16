import cv2
import numpy as np

import pathlib
LOCAL = pathlib.Path(__file__).parent
image = cv2.imread(LOCAL / "w12images" / "WhatsApp Image 2024-11-15 at 17.38.09 (1).jpeg")

images = [ cv2.imread(LOCAL / "w12images" / a) for a in [
	"WhatsApp Image 2024-11-15 at 17.38.35 (4).jpeg",
	"WhatsApp Image 2024-11-15 at 17.38.09.jpeg",
	"WhatsApp Image 2024-11-15 at 17.38.49 (2).jpeg"
] ]

# cv2.imshow("Original image", image)

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

for i, img in enumerate(images):
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# -> Grayscale
	lines_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Find lines via threshold
	thresh = cv2.adaptiveThreshold(lines_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 15)
	lines_image[thresh == 255] = 255
	lines_image = cv2.bitwise_not(lines_image)
	cv2.imshow(f"Contrast{i}", cv2.resize(lines_image, None, fx=1/2, fy=1/2))
	
	lines = cv2.HoughLinesP(lines_image, 1, np.pi/180, 1, np.array([]), 150, 2)
	lines_image = cv2.cvtColor(lines_image, cv2.COLOR_GRAY2BGR)
	for line in lines:
		for x1, y1, x2, y2 in line:
			cv2.line(lines_image, (x1, y1), (x2, y2), (255, 255, 0), 2)

	cv2.imshow(f"Detected Lines{i}", lines_image)
cv2.waitKey(0)

# Find lines

# cv2.imwrite(LOCAL / "out.png", lines_image)