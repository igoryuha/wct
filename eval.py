import torch
from models import NormalisedVGG, Decoder
import argparse
import os


parser = argparse.ArgumentParser(description='WCT')
parser.add_argument('--content-path', type=str, required=True)
parser.add_argument('--style-path', type=str, required=True)
parser.add_argument('--encoder-path', type=str, default='encoder/vgg_normalised_conv5_1.pth')
parser.add_argument('--decoders-dir', type=str, default='decoders')
parser.add_argument('--style-decorator', type=bool, default=True)
parser.add_argument('--kernel-size', type=int, default=7)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--alpha', type=int, default=0.8)
parser.add_argument('--ss-alpha', type=int, default=0.6)
parser.add_argument('--gpu', type=str, default=0)

args = parser.parse_args()

device = torch.device('cuda:%s' % args.gpu if torch.cuda.is_available() else 'cpu')

encoder = NormalisedVGG(pretrained_path=args.encoder_path).to(device)
d5 = Decoder('relu5_1', pretrained_path=os.path.join(args.decoders_dir, 'd5.pth')).to(device)
d4 = Decoder('relu4_1', pretrained_path=os.path.join(args.decoders_dir, 'd4.pth')).to(device)
d3 = Decoder('relu3_1', pretrained_path=os.path.join(args.decoders_dir, 'd3.pth')).to(device)
d2 = Decoder('relu2_1', pretrained_path=os.path.join(args.decoders_dir, 'd2.pth')).to(device)
d1 = Decoder('relu1_1', pretrained_path=os.path.join(args.decoders_dir, 'd1.pth')).to(device)

