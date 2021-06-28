import cv2
import matplotlib.pyplot as plt

# plots a small circle - use left click for a filled circle or right click for a donut
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, center = (x, y), radius = 20, 
                       color = (255, 0, 237), thickness = -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.circle(img, center = (x, y), radius = 20,  
                       color = (255, 0, 237), thickness = 5)

drawing = False
ix = -1
iy = -1

# creates a rectangle from the point you clicked on the map
# whilst holding down the mouse it continues to draw
def draw_rectangle(event, x, y, flags, params):
    
    global ix, iy, drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(img, pt1=(ix, iy), pt2=(x, y), 
                          color = (94, 196, 224), thickness = -1)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, pt1=(ix, iy), pt2=(x, y), 
                     color = (94, 196, 224), thickness = -1)

img = cv2.imread("assets/game-of-thrones-map.jpeg")
img = cv2.resize(img, (1200, 800))
win_name = "my_map"
cv2.namedWindow(winname = win_name)
cv2.setMouseCallback(win_name, draw_rectangle)


while True:
    # note that the imshow has to be within this loop for the circle drawing update to work
    cv2.imshow(win_name, img)
    # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1/39201163
    # press ESC to close
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
# Once script is done, its usually good practice to call this line
# It closes all windows (just in case you have multiple windows called)
cv2.destroyAllWindows()

