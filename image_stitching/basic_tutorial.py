import cv2
import os
import json

config = json.load(open('/Users/emiliolr/Desktop/counting-cranes/config.json', 'r'))
CODE_FP = config['code_filepath_local']

image_fps = sorted(os.listdir(os.path.join(CODE_FP, 'image_stitching', 'test_images', 'uninterrupted_flightline')))
image_fps = [fp for fp in image_fps if fp.startswith('FLIR')]
print(len(image_fps))
images = [cv2.imread(os.path.join('test_images', 'uninterrupted_flightline', fp)) for fp in image_fps]
# cv2.imshow('TEST', images[0])
# cv2.waitKey(0)

#DUMP METHOD: just handing all images to openCV to stitch, but often leaves out images in flight line
stitcher = cv2.Stitcher_create()
status, stitched_img = stitcher.stitch(images)
print(status)
if status == 0:
    cv2.imwrite('stitched_parent_image_TEST.png', stitched_img)
    cv2.imshow('Stitched Image', stitched_img)
    cv2.waitKey(0)

#ITERATIVE METHOD: moving along the flight line and adding in one parent image at a time
# images_to_stitch = []
# stitcher = cv2.Stitcher_create()
# status, stitched_img = stitcher.stitch(images[ : 2])
# cv2.imshow('Stitched Image', stitched_img)
# cv2.waitKey(0)
# images_to_stitch.append(stitched_img)
# images_to_stitch.extend(images[2 : 4])
# status, stitched_img = stitcher.stitch(images_to_stitch)
# print(status)
# cv2.imshow('Stitched Image', stitched_img)
# cv2.waitKey(0)
