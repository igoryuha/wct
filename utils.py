import torch
from torchvision import transforms
from ops import relu_x_1_style_decorator_transform, relu_x_1_transform
from PIL import Image


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


def forward_transform(
        device, save_path, content_path, style_path, encoder, decoders, content_size,
        style_size, alpha, decorator, kernel_size=7, stride=1, ss_alpha=0.6
):

    content = preprocess(load_image(content_path), content_size).to(device)
    style = preprocess(load_image(style_path), style_size).to(device)

    d5, d4, d3, d2, d1 = decoders

    if decorator:
        relu5_1_recons = relu_x_1_style_decorator_transform(content, style, encoder, d5,
                                                            'relu5_1', kernel_size, stride, ss_alpha)
    else:
        relu5_1_recons = relu_x_1_transform(content, style, encoder, d5, 'relu5_1', alpha)
    print('relu5_1 done')
    relu4_1_recons = relu_x_1_transform(relu5_1_recons, style, encoder, d4, 'relu4_1', alpha)
    print('relu4_1 done')
    relu3_1_recons = relu_x_1_transform(relu4_1_recons, style, encoder, d3, 'relu3_1', alpha)
    print('relu3_1 done')
    relu2_1_recons = relu_x_1_transform(relu3_1_recons, style, encoder, d2, 'relu2_1', alpha)
    print('relu2_1 done')
    relu1_1_recons = relu_x_1_transform(relu2_1_recons, style, encoder, d1, 'relu1_1', alpha)
    print('relu1_1 done')

    deprocess(relu1_1_recons).save(save_path)
