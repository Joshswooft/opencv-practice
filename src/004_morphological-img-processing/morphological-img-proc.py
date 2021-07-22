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



# 0 = wait for infinite time for any key press
cv2.waitKey(0)
cv2.destroyAllWindows()