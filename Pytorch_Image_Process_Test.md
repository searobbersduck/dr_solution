# 如何利用训练好的模型对单张图像进行分类

## 1. 测试如何用pytorch进行图像处理

主要用到torchvision的transforms中的方法。

如果没有安装torchvision，请参考如下链接进行安装：
[torchvision](https://github.com/pytorch/vision)

```
import torchvision.transforms as transforms
```

torchvision.transforms中有各种现成的方法可以用：
```python
Normalize
Scale
CenterCrop
Pad
Lambda
RandomCrop
RandomHorizontalFlip
RandomSizedCrop
``` 
简单的处理如下：
```python
from PIL import Image
import torchvision.transforms as transforms

img = Image.open('../sample/3/16_left.jpeg')

img.show()

trans = transforms.Compose([
    transforms.Scale(512),
    transforms.RandomCrop(448)
])

img1 = trans(img)
img1.show()
```

上述代码，是利用pytorch torchvision中的transforms对图像进行处理的结果。

![原始图像](./test/raw.png)
![处理图像](./test/img1.png)


其中的```ToTensor```和```ToPILImage```可以支持image和tensor数据的相互转换，【0.0，1.0】到【0，255】的数据范围转换，相应的也会有通道位置的调整。

有了上述前提，可以将图片转换为pytorch需要的tensor结构，也可以从pytorch的tensor结构转换为图像进行显示。

## 2. 
