import glob
from PIL import Image
import numpy as np
import os


def resizer(counter, img_path, save_path, save_path_ending):
    image_reference = Image.open(img_path)
    print(counter)

    resized_image = image_reference.resize((256,256))
    resized_image.save(save_path + '_' + str(counter) + '_' + save_path_ending)


# Constructs a list of PNG image names in raw_images
# raw_images = glob.glob('C:\\Users\\zuhat\\Downloads\\heightmaps\\*.png')
raw_images = glob.glob('C:\\heightmaps\\use_these\\coast_dataset\\coasts\\*.png')

counter = 0
save_path = 'C:\\heightmaps\\use_these\\resized\\coasts\\'
save_path_ending = '.png'

for raw_image in raw_images:
    resizer(counter, raw_image, save_path, save_path_ending)
    counter = counter + 1
