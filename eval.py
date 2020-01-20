import torch
from models import NormalisedVGG, Decoder
from utils import forward_transform
import argparse
import os


parser = argparse.ArgumentParser(description='WCT')
parser.add_argument('--content-path', type=str, required=True)
parser.add_argument('--style-path', type=str, required=True)
parser.add_argument('--save-path', type=str, default='./result.jpg')
parser.add_argument('--encoder-path', type=str, default='encoder/vgg_normalised_conv5_1.pth')
parser.add_argument('--decoders-dir', type=str, default='decoders')
parser.add_argument('--content-size', type=int, default=768, help='smaller side size of content image')
parser.add_argument('--style-size', type=int, default=768, help='smaller side size of style image')
parser.add_argument('--style-decorator', type=int, default=1)
parser.add_argument('--kernel-size', type=int, default=7)
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

with torch.no_grad():
    forward_transform(device, args.save_path, args.content_path, args.style_path, encoder, [d5, d4, d3, d2, d1],
                      args.content_size, args.style_size, args.alpha, args.style_decorator,
                      args.kernel_size, args.stride, args.ss_alpha)
