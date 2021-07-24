import cv2
import numpy as np
import matplotlib.pyplot as plt


img_path = "assets/004_morphological/tut/"
img_name = "rick_morty.jpg"
img = cv2.imread(img_path + img_name)

print("hello")
cv2.imshow("rick and morty", img)
img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray", gray)


# Eroded example

kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(gray,kernel,iterations = 1)
cv2.imshow("eroded", erosion)
cv2.imwrite(img_path + "eroded.png", erosion)

# dilation example - increases white area

dilation = cv2.dilate(gray, kernel, iterations = 1)
cv2.imshow("dilation", dilation)
cv2.imwrite(img_path + "dilation.png", dilation)

# Opening

opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
cv2.imshow("opening", opening)
cv2.imwrite(img_path + "opening.png", opening)

# closing

closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
cv2.imshow("closing", closing)
cv2.imwrite(img_path + "closing.png", closing)

# using structured elements for kernel

se = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
closing_se = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)

cv2.imshow("closing with se", closing_se)

# Boundary Extraction

ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
cv2.imshow('Binary image', thresh)

# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

# draw contours on the original image
image_copy = img.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
cv2.imshow('None approximation', image_copy)

# Filling holes

im_flood = thresh.copy()

# Notice the size needs to be 2 pixels than the image.
h, w = thresh.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_flood, mask, (0,0), 255)

# invert
im_floodfill_inv = cv2.bitwise_not(im_flood)

# Combine the two images to get the foreground.
im_out = thresh | im_floodfill_inv

cv2.imshow("Flood", im_flood)
cv2.imshow("Inverted flood", im_floodfill_inv)
cv2.imshow("Foreground", im_out)

# 0 = wait for infinite time for any key press
cv2.waitKey(0)
cv2.destroyAllWindows()