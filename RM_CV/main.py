import cv2
import argparse
# import digit_recognize as dr
import armor_detect as ad

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",default='images/img3.jpg', help = "path to the image file")
ap.add_argument("-v","--video", help = "path to the video file")
args = vars(ap.parse_args())

if __name__ == '__main__':
    frame = cv2.imread(args["image"])
    (hsv,gray,binary) = ad.get_color(frame,1)
    ad.detect_armor_red_image(frame,gray,binary)
    (hsv,gray,binary) = ad.get_color(frame,2)
    ad.detect_armor_blue_image(frame,gray,binary)