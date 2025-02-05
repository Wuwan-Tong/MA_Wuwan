import cv2
import os


# RGB image dir
img_dir_rgb="C:/Users/INDA_HIWI/Desktop/flickr30k/flickr30k-images"
assert os.path.exists(img_dir_rgb), 'rgb image dir not exist'
# Gray image dir
img_dir_gray="C:/Users/INDA_HIWI/Desktop/flickr30k/flickr30k-images_gray"
if not os.path.exists(img_dir_gray):
    os.makedirs(img_dir_gray)

for img_filename in os.listdir(img_dir_rgb):
    # RGB image path
    img_path_rgb = os.path.join(img_dir_rgb, img_filename)
    # read rgb images in mode gray
    image = cv2.imread(img_path_rgb, cv2.IMREAD_GRAYSCALE)
    # Gray image path
    img_path_gray = os.path.join(img_dir_gray, img_filename)
    # save gray images
    cv2.imwrite(img_path_gray,image)