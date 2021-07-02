import cv2
import numpy as np

print(cv2.__version__)

base_path_to_save = "assets/spatial_images/"
img_path = "assets/surfing.jpg"

# img negative
img = cv2.imread(img_path)
img_thumb = cv2.resize(img, (200, 100))
cv2.imwrite(base_path_to_save + "surfing_thumb.png", img_thumb)
gray = cv2.imread(img_path, 0)
gray = cv2.resize(gray, (200, 100))

cv2.imshow("Grayscale image", gray)
cv2.imwrite(base_path_to_save + "grayscale.png", gray)

L = gray.max()

# Maximum grey level value  minus 
# the original image gives the
# negative image
img_neg = L - gray

cv2.imshow("Negative image", img_neg)
cv2.imwrite(base_path_to_save + "negative.jpg", img_neg)

def thresholding(img, threshold = 150):
    # create a array of zeros
    m,n = gray.shape
    img_thresh = np.zeros((m,n), dtype = int) 
    
    for i in range(m):
        
        for j in range(n):
            
            if img[i,j] < threshold: 
                img_thresh[i,j]= 0
            else:
                img_thresh[i,j] = 255
    return img_thresh


thresholded_img = thresholding(gray)
thresh_path = base_path_to_save + "thresholded.png"
cv2.imwrite(thresh_path, thresholded_img)
cv2.imshow("thresholded", cv2.imread(thresh_path))

# 0 = wait for infinite time for any key press
cv2.waitKey(0)
cv2.destroyAllWindows()

