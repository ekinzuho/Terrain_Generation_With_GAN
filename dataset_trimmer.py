import glob
from PIL import Image
import numpy as np
import os

# Constructs a list of PNG image names in raw_images
#raw_images = glob.glob('C:\\Users\\zuhat\\Downloads\\heightmaps\\*.png')
raw_images = glob.glob('C:\\heightmaps\\new_heightmaps\\*.png')

# This loop checks and filters the heightmaps that has a non-sufficient standard deviation value.
for raw_image in raw_images:
    print(raw_image)

    # getting pixel information from the PNG in current loop
    pixel_data = Image.open(raw_image).getdata()

    # calculating a standard deviation value to filter all black or all white pictures.
    standard_deviation = np.sqrt(np.var(pixel_data))
    # print(standard_deviation)

    """
    # below is our latest criteria to trim out dataset
    # we started trimming from <500 and eventually came to the conclusion
    # that the value threshold below is sufficient for our purpose
    # here we delete them using os.remove()
    """
    if standard_deviation < 3000:
        os.remove(raw_image)
