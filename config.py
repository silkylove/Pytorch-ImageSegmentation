# -*- coding: utf-8 -*-
import os
import logging
import argparse

unet_params = {'filter_scale': 1}

deeplabv3_params = {'backend': 'resnet101',
                    'os': 16,
                    'pretrained': True}

pspnet_params = {'backend': 'resnet18',
                 'pool_scales': (1, 2, 3, 6),
                 'pretrained': True}

ce_params = {'weight': None, 'ignore_index': 255}
dice_params = {'smooth': 1}
focal_params = {'weight': None, 'gamma': 2, 'alpha': 0.5}
lovasz_params = {'multiclasses': True}

parse = argparse.ArgumentParser(description='ImageSegmentation')

parse.add_argument('--model_params', default={'unet': unet_params,
                                              'deeplabv3+': deeplabv3_params,
                                              'pspnet': pspnet_params})

parse.add_argument('--loss_params', default={'ce': ce_params,
                                             'dice': dice_params,
                                             'focal': focal_params,
                                             'lovasz': lovasz_params})

parse.add_argument('--model', default='deeplabv3+', choices=['unet', 'deeplabv3+', 'pspnet'], type=str)
parse.add_argument('--loss', default='ce', choices=['ce', 'dice', 'focal', 'lovasz'], type=str)
parse.add_argument('--lr', default=1e-2, type=float)
# parse.add_argument('--lr_decay_step', default=[30, 40], type=list)
# parse.add_argument('--lr_decay_rate', default=0.1, type=float)
parse.add_argument('--max_iters', default=30000, type=int)
parse.add_argument('--epochs', default=None)
parse.add_argument('--batch_size', default=16 * 1, type=int)
parse.add_argument('--distributed', default=True, type=bool)
parse.add_argument('--gpuid', default='0,1,2,3', type=str)
parse.add_argument('--num_workers', default=8, type=int)
parse.add_argument('--ckpt_dir', default='./checkpoint_1/')
parse.add_argument('--resume', default=False, help='resume from checkpoint', type=bool)

parse.add_argument('--image_size', default=513, type=int)
parse.add_argument('--in_channels', default=3, type=int)
parse.add_argument('--image_root', default='/home/yhuangcc/data/VOC2012/')
parse.add_argument('--train_list', default='/home/yhuangcc/data/VOC2012/list/train_aug.txt')
parse.add_argument('--val_list', default='/home/yhuangcc/data/VOC2012/list/val.txt')
parse.add_argument('--label_file', default='/home/yhuangcc/ImageSegmentation/datasets/voc/labels')

log_dir = './log_1/'
parse.add_argument('--log_dir', default=log_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logger = logging.getLogger("InfoLog")
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(log_dir + 'log.txt')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(console)


def get_config():
    config, unparsed = parse.parse_known_args()
    return config
