import os
import cv2

ds_ratio= 1/2 # down-sampling ratio
# original image dir
img_dir_ori="C:/Users/INDA_HIWI/Desktop/flickr30k/flickr30k-images"
assert os.path.exists(img_dir_ori), 'original image dir not exist'
# down-sampled image dir
img_dir_ds="C:/Users/INDA_HIWI/Desktop/flickr30k/flickr30k-images_ds32"
if not os.path.exists(img_dir_ds):
    os.makedirs(img_dir_ds)

for img_filename in os.listdir(img_dir_ori):
    # original image path
    img_path_ori = os.path.join(img_dir_ori, img_filename)
    # load original image
    image = cv2.imread(img_path_ori)
    # down-sampling
    image_ds = cv2.resize(image, None, fx=ds_ratio, fy=ds_ratio, interpolation=cv2.INTER_CUBIC)
    # down-sampled image path
    img_path_ds = os.path.join(img_dir_ds, img_filename)
    # save down-sampled image
    cv2.imwrite(img_path_ds, image_ds)
