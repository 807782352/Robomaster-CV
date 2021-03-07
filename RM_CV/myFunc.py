'''
    Here is to store any useful functions that might be used in image processing.
'''

import cv2

def cv_show(name,img):
   cv2.imshow(name, img)
   cv2.waitKey(0)
   cv2.destroyAllWindows()