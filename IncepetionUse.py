import torch
from torchvision import transforms, datasets
from IncepetionLearning import InceptionV3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

import matplotlib.pyplot as plt
from PIL import Image

train_transforms = transforms.Compose([
    transforms.Resize([224, 224]),  # 将输入图片resize成统一尺寸
    # transforms.RandomHorizontalFlip(), # 随机水平翻转
    transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
    transforms.Normalize(  # 标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])  # 其中 mean=[0.485,0.456,0.406]与std=[0.229,0.224,0.225] 从数据集中随机抽样计算得到的。
])

test_transform = transforms.Compose([
    transforms.Resize([224, 224]),  # 将输入图片resize成统一尺寸
    transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
    transforms.Normalize(  # 标准化处理-->转换为标准正太分布（高斯分布），使模型更容易收敛
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])  # 其中 mean=[0.485,0.456,0.406]与std=[0.229,0.224,0.225] 从数据集中随机抽样计算得到的。
])

# 这是 PyTorch 的一个工具，用于加载目录结构化的数据集。它会将目录中的图像按照子文件夹的名称自动分配标签。每个子文件夹代表一个类别，子文件夹内的所有图像都属于该类别。
total_data = datasets.ImageFolder("./data/images", transform=train_transforms)
#class_to_idx 是一个属性，它是一个字典，表示类别名称到类别索引之间的映射关系。例如，假设你有两个类别 class1 和 class2，则 class_to_idx 可能是 { 'class1': 0, 'class2': 1 }。
print(total_data.class_to_idx)

classes = list(total_data.class_to_idx)
print(classes)

def predict_one_image(image_path, model, transform, classes):
    test_img = Image.open(image_path).convert('RGB')
    plt.imshow(test_img)  # 展示预测的图片

    test_img = transform(test_img)
    img = test_img.to(device).unsqueeze(0)

    model.eval()
    output = model(img)

    _, pred = torch.max(output, 1)
    print(pred)
    pred_class = classes[pred]
    print(f'预测结果是：{pred_class}')

# 调用并将模型转移到GPU中
model = InceptionV3().to(device)
model.load_state_dict(torch.load('./best_model.pth',map_location=device))

# 预测训练集中的某张照片
predict_one_image(image_path='./data/images/american/american_bulldog_2.jpg',
                  model=model,
                  transform=train_transforms,
                  classes=classes)