import IPython
from IPython import display
import imageio
import glob

directory = 'C:\\GAN\\3D-Terrain-Generation-GAN\\gif_source'
gif = 'C:\\GAN\\3D-Terrain-Generation-GAN\\gif\\example.gif'

with imageio.get_writer(gif, mode='I') as writer:
    filenames = glob.glob(directory + '\\*.png')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
        frame = 2 * (i ** 0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)

if IPython.version_info > (6, 2, 0, ''):
    display.Image(filename=gif)

display.Image(filename=gif)
