import torch
from models import NormalisedVGG, Decoder
from utils import load_image, preprocess, deprocess, extract_images
from ops import style_decorator, wct
import argparse
import os


parser = argparse.ArgumentParser(description='WCT')

parser.add_argument('--content-path', type=str, help='path to the content image')
parser.add_argument('--style-path', type=str, help='path to the style image')
parser.add_argument('--content-dir', type=str, help='path to the content image folder')
parser.add_argument('--style-dir', type=str, help='path to the style image folder')

parser.add_argument('--save-name', type=str, default='result')
parser.add_argument('--save-dir', type=str, default='./results')
parser.add_argument('--save-ext', type=str, default='jpg', help='The extension name of the output image')
parser.add_argument('--encoder-path', type=str, default='encoder/vgg_normalised_conv5_1.pth')
parser.add_argument('--decoders-dir', type=str, default='decoders')
parser.add_argument('--content-size', type=int, default=768, help='New (minimum) size for the content image')
parser.add_argument('--style-size', type=int, default=768, help='New (minimum) size for the style image')

parser.add_argument('--style-decorator', type=int, default=1)
parser.add_argument('--kernel-size', type=int, default=12)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.8)
parser.add_argument('--ss-alpha', type=float, default=0.6)

parser.add_argument('--use-gpu', type=int, default=1)
parser.add_argument('--gpu', type=str, default=0)

args = parser.parse_args()

device = torch.device('cuda:%s' % args.gpu if torch.cuda.is_available() and args.use_gpu else 'cpu')

encoder = NormalisedVGG(pretrained_path=args.encoder_path).to(device)
d5 = Decoder('relu5_1', pretrained_path=os.path.join(args.decoders_dir, 'd5.pth')).to(device)
d4 = Decoder('relu4_1', pretrained_path=os.path.join(args.decoders_dir, 'd4.pth')).to(device)
d3 = Decoder('relu3_1', pretrained_path=os.path.join(args.decoders_dir, 'd3.pth')).to(device)
d2 = Decoder('relu2_1', pretrained_path=os.path.join(args.decoders_dir, 'd2.pth')).to(device)
d1 = Decoder('relu1_1', pretrained_path=os.path.join(args.decoders_dir, 'd1.pth')).to(device)


def style_transfer(content, style):

    if args.style_decorator:
        relu5_1_cf = encoder(content, 'relu5_1')
        relu5_1_sf = encoder(style, 'relu5_1')
        relu5_1_scf = style_decorator(relu5_1_cf, relu5_1_sf, args.kernel_size, args.stride, args.ss_alpha)
        relu5_1_recons = d5(relu5_1_scf)
    else:
        relu5_1_cf = encoder(content, 'relu5_1')
        relu5_1_sf = encoder(style, 'relu5_1')
        relu5_1_scf = wct(relu5_1_cf, relu5_1_sf, args.alpha)
        relu5_1_recons = d5(relu5_1_scf)

    relu4_1_cf = encoder(relu5_1_recons, 'relu4_1')
    relu4_1_sf = encoder(style, 'relu4_1')
    relu4_1_scf = wct(relu4_1_cf, relu4_1_sf, args.alpha)
    relu4_1_recons = d4(relu4_1_scf)

    relu3_1_cf = encoder(relu4_1_recons, 'relu3_1')
    relu3_1_sf = encoder(style, 'relu3_1')
    relu3_1_scf = wct(relu3_1_cf, relu3_1_sf, args.alpha)
    relu3_1_recons = d3(relu3_1_scf)

    relu2_1_cf = encoder(relu3_1_recons, 'relu2_1')
    relu2_1_sf = encoder(style, 'relu2_1')
    relu2_1_scf = wct(relu2_1_cf, relu2_1_sf, args.alpha)
    relu2_1_recons = d2(relu2_1_scf)

    relu1_1_cf = encoder(relu2_1_recons, 'relu1_1')
    relu1_1_sf = encoder(style, 'relu1_1')
    relu1_1_scf = wct(relu1_1_cf, relu1_1_sf, args.alpha)
    relu1_1_recons = d1(relu1_1_scf)

    return relu1_1_recons


if not os.path.exists(args.save_dir):
    print('Creating save folder at', args.save_dir)
    os.mkdir(args.save_dir)

with torch.no_grad():

    if args.content_dir and args.style_dir:
        content_paths = extract_images(args.content_dir)
        style_paths = extract_images(args.style_dir)

        for i in range(len(content_paths)):
            content = load_image(content_paths[i])
            content = preprocess(content, args.content_size)
            content = content.to(device)

            for j in range(len(style_paths)):
                style = load_image(style_paths[j])
                style = preprocess(style, args.style_size)
                style = style.to(device)

                output = style_transfer(content, style)
                output = deprocess(output)
                save_path = '%s/%s_%s.%s' % (args.save_dir, i, j, args.save_ext)
                print('Output image saved at:', save_path)
                output.save(save_path)
    else:
        content = load_image(args.content_path)
        content = preprocess(content, args.content_size)
        content = content.to(device)

        style = load_image(args.style_path)
        style = preprocess(style, args.style_size)
        style = style.to(device)

        output = style_transfer(content, style)
        output = deprocess(output)
        save_path = '%s/%s.%s' % (args.save_dir, args.save_name, args.save_ext)
        print('Output image saved at:', save_path)
        output.save(save_path)
