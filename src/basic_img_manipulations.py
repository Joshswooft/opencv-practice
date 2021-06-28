import cv2
import matplotlib.pyplot as plt

print(cv2.__version__)

img_path = "assets/surfing.jpg"
# opencv by default loads images in BGR (blue, green, red) rather than standard RGB
img = cv2.imread(img_path)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gs = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
cv2.imshow("RGB", rgb_img)
cv2.imshow("Grayscale", img_gs)

# plt each R, G, B seperately
fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 20))
for i in range(0, 3):
    ax = axs[i]
    ax.imshow(rgb_img[:, :, i], cmap = 'gray')
plt.show()

# Transform the image into HSV and HLS models
# HSV = Hue, saturation and value, 3D representations of the colour model
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# HSL = Hue, Saturation and Lightness. 
img_hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
cv2.imshow("HSV", img_hsv)
cv2.imshow("HLS", img_hsl)

# 100 x 100
resized_img = cv2.resize(img, (100, 100))
cv2.imshow("my image", img)
# half the size and width
resized_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
cv2.imshow("thumbnail", resized_img)

rotated_img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow("90cc", rotated_img)

# save an img
file_to_save = "assets/surfing_gs.jpg"
cv2.imwrite(file_to_save, img_gs)

# 0 = wait for infinite time for any key press
cv2.waitKey(0)
cv2.destroyAllWindows()
