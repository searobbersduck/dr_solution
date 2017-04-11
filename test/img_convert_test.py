from PIL import Image
import torchvision.transforms as transforms

img = Image.open('../sample/3/16_left.jpeg')

img.show()

# img.save('./raw.png')

trans1 = transforms.Compose([
    transforms.Scale(512),
    transforms.RandomCrop(448)
])

img1 = trans1(img)
img1.show()

# img1.save('./img1.png')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
trans2 = transforms.Compose([
    transforms.Scale(512),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    normalize,
    transforms.ToPILImage()
])

img2 = trans2(img)
img2.show()

# img2.save('./img2.png')

