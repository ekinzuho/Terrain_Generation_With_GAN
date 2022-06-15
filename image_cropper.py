import glob
from PIL import Image
import numpy as np
import os


def crop_and_save(counter, img_path, save_path, save_path_ending):
    image_to_crop = Image.open(img_path)

    # below are values that allowed me to crop 1081x1081 into 4 512x512 files
    """
    topleft = (0,0,512,512)
    topright = (512,0,1024,512)
    bottomleft = (0,512,512,1024)
    bottomright = (512,512,1024,1024)
    """

    # below are the values that allowed me to break 512x512's into 4 256x256 files
    topleft = (0,0,256,256)
    topright = (256,0,512,256)
    bottomleft = (0,256,256,512)
    bottomright = (256,256,512,512)


    print(counter)

    image_to_crop.crop(topleft).save(save_path + str(counter) + save_path_ending, quality=100)
    counter = counter + 1
    print(counter)

    image_to_crop.crop(topright).save(save_path + str(counter) + save_path_ending, quality=100)
    counter = counter + 1
    print(counter)

    image_to_crop.crop(bottomleft).save(save_path + str(counter) + save_path_ending, quality=100)
    counter = counter + 1
    print(counter)

    image_to_crop.crop(bottomright).save(save_path + str(counter) + save_path_ending, quality=100)


# Constructs a list of PNG image names in raw_images
# raw_images = glob.glob('C:\\Users\\zuhat\\Downloads\\heightmaps\\*.png')
raw_images = glob.glob('C:\\Users\\zuhat\\Downloads\\heightmaps\\cropped\\*.png')

counter = 0
save_path = 'C:\\Users\\zuhat\\Downloads\\heightmaps\\cropped\\further-cropped\\cropped_heightmap'
save_path_ending = '.png'

for raw_image in raw_images:
    crop_and_save(counter, raw_image, save_path, save_path_ending)
    counter = counter + 4
