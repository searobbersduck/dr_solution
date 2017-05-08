from PIL import Image
import torchvision.transforms as transforms

img = Image.open('../sample/3/16_left.jpeg')

# img.show()

# img.save('./raw.png')

trans1 = transforms.Compose([
    transforms.Scale(512),
    transforms.RandomCrop(448)
])

img1 = trans1(img)
# img1.show()

# img1.save('./img1.png')

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])

MEAN = [108.64628601, 75.86886597, 54.34005737]
STD = [70.53946096, 51.71475228, 43.03428563]

MEAN = [x/255 for x in MEAN]
STD = [x/255 for x in STD]

print(MEAN)
print(STD)

normalize = transforms.Normalize(mean=MEAN,
                                 std=STD)



trans2 = transforms.Compose([
    transforms.Scale(512),
    transforms.ToTensor(),
    normalize,
    transforms.ToPILImage()
])

img2 = trans2(img)
img2.show()

# img2.save('./img2.png')



# # tight crop
# import numpy as np
# import scipy.misc
# from skimage.filters import threshold_otsu
# from skimage import measure, exposure
#
# import matplotlib.pyplot as plt
#
# def tight_crop(img, size=None):
#     img_gray = np.mean(img,2)
#     img_bw = img_gray > threshold_otsu(img_gray)
#     img_label = measure.label(img_bw, background=0)
#     largest_label = np.argmax(np.bincount(img_label.flatten())[1:])+1
#     img_circ = (img_label == largest_label)
#     img_xs = np.sum(img_circ,0)
#     img_ys = np.sum(img_circ,1)
#     xs = np.where(img_xs>0)
#     ys = np.where(img_ys>0)
#     y_lo = np.min(ys)
#     y_hi = np.max(ys)
#     x_lo = np.min(xs)
#     x_hi = np.max(xs)
#     img_crop = img[y_lo:y_hi, x_lo:x_hi, :]
#     return img_crop
#
# path_img = './raw.png'
#
# img = scipy.misc.imread(path_img)
# img = img.astype(np.float32)
# img /= 255
#
# img_crop = tight_crop(img)
#
#
# plt.rcParams['figure.figsize'] = [8,8]
# fig, ax = plt.subplots(1,2)
# ax[0].imshow(img)
# ax[0].axis('off')
# ax[0].set_title('Original')
# ax[1].imshow(img_crop)
# ax[1].axis('off')
# ax[1].set_title('Tight Crop')
#
#
# def channelwise_ahe(img):
#     img_ahe = img.copy()
#     for i in range(img.shape[2]):
#         img_ahe[:,:,i] = exposure.equalize_adapthist(img[:,:,i], clip_limit=0.03)
#     return img_ahe
#
# img_ahe = channelwise_ahe(img_crop)
#
# fig, ax = plt.subplots(1,2)
# ax[0].imshow(img_crop)
# ax[0].axis('off')
# ax[0].set_title('Original')
# ax[1].imshow(img_ahe)
# ax[1].axis('off')
# ax[1].set_title('High Contrast')