import cv2
import numpy as np
import pyautogui as gui
from time import time

loop_time = time()

while(True):

    screenshot = gui.screenshot()
    # re-shape into format opencv understands
    screenshot = np.array(screenshot)

    # remember opencv uses BGR so we have to convert
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

    cv2.imshow("Screenshot", screenshot)

    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break