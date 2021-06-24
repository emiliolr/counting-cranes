import cv2
import os
import json

config = json.load(open('/Users/emiliolr/Desktop/counting-cranes/config.json', 'r'))
CODE_FP = config['code_filepath_local']

image_fps = sorted(os.listdir(os.path.join(CODE_FP, 'image_stitching', 'test_images')))
image_fps = [fp for fp in image_fps if fp.startswith('test_thermal')]
print(image_fps)
images = [cv2.imread(os.path.join('test_images', fp)) for fp in image_fps]
# cv2.imshow('TEST', images[1])
# cv2.waitKey(0)

stitcher = cv2.Stitcher_create()
status, stitched_img = stitcher.stitch(images)
print(status)
cv2.imshow('Stitched Image', stitched_img)
cv2.waitKey(0)
