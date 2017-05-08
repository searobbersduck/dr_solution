import numpy as np

from skimage.filters import threshold_otsu
from skimage import measure, exposure

import skimage

import scipy.misc
from PIL import Image

def tight_crop(img, size=None):
    img_gray = np.mean(img, 2)
    img_bw = img_gray > threshold_otsu(img_gray)
    img_label = measure.label(img_bw, background=0)
    largest_label = np.argmax(np.bincount(img_label.flatten())[1:])+1

    img_circ = (img_label == largest_label)
    img_xs = np.sum(img_circ, 0)
    img_ys = np.sum(img_circ, 1)
    xs = np.where(img_xs>0)
    ys = np.where(img_ys>0)
    x_lo = np.min(xs)
    x_hi = np.max(xs)
    y_lo = np.min(ys)
    y_hi = np.max(ys)
    img_crop = img[y_lo:y_hi, x_lo:x_hi, :]

    return img_crop



# img = scipy.misc.imread('./raw1.jpg')
img = scipy.misc.imread('./raw.png')

img = img.astype(np.float32)
img /= 255

img_crop = tight_crop(img)


pilImage = Image.fromarray(skimage.util.img_as_ubyte(img_crop))
# pilImage.show()


# adaptive historgram equlization
def channelwise_ahe(img):
    img_ahe = img.copy()
    for i in range(img.shape[2]):
        img_ahe[:,:,i] = exposure.equalize_adapthist(img[:,:,i], clip_limit=0.03)
    return img_ahe

img_ahe = channelwise_ahe(img_crop)

pilImage = Image.fromarray(skimage.util.img_as_ubyte(img_ahe))
pilImage.show()

mean = np.array([108.64628601, 75.86886597, 54.34005737])
std = np.array([70.53946096, 51.71475228, 43.03428563])

# subtract the local average
img = scipy.misc.imread('./raw.png')
img = img.astype(np.float32)
# img -= np.mean(img, (0,1))
img -= mean
img /=std

print(np.max(img))

print(np.mean(img, (0,1)))

pilImage = Image.fromarray(skimage.util.img_as_ubyte(img))

pilImage.show()