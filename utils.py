import torch
from torchvision import transforms
from ops import relu_x_1_style_decorator_transform, relu_x_1_transform
from PIL import Image
import os


def eval_transform(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])


def load_image(path):
    return Image.open(path).convert('RGB')


def preprocess(img, size):
    transform = eval_transform(size)
    return transform(img).unsqueeze(0)


def deprocess(tensor):
    tensor = tensor.cpu()
    tensor = tensor.squeeze(0)
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor)


def extract_image_names(path):
    r_ = []
    valid_ext = ['.jpg', '.png']

    items = os.listdir(path)

    for item in items:
        item_path = os.path.join(path, item)

        _, ext = os.path.splitext(item_path)
        if ext not in valid_ext:
            continue

        r_.append(item_path)

    return r_
