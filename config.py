# -*- coding: utf-8 -*-
import os
import logging
import argparse

'''
All backbones
['resnet18', 'resnet34', 'resnet50', 'resnet101',
 'resnet152', 'senet154', 'se_resnet50', 'se_resnet101',
 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d',
 'mobilenet_v2', 'shufflenet_v2'(for deeplabv3+)]
'''

unet_params = {'filter_scale': 1}

unet_ae_params = {'backend': 'resnet101',
                  'pretrained': 'imagenet'}

deeplabv3_params = {'backend': 'mobilenet_v2',
                    'os': 16,
                    'pretrained': 'imagenet'}

pspnet_params = {'backend': 'resnet101',
                 'pool_scales': (1, 2, 3, 6),
                 'pretrained': 'imagenet'}

ce_params = {'weight': None, 'ignore_index': 255}
dice_params = {'smooth': 1}
focal_params = {'weight': None, 'gamma': 2, 'alpha': 0.5}
lovasz_params = {'multiclasses': True}

parse = argparse.ArgumentParser(description='ImageSegmentation')

parse.add_argument('--model_params', default={'unet': unet_params,
                                              'unet_ae': unet_ae_params,
                                              'dlv3plus': deeplabv3_params,
                                              'pspnet': pspnet_params})

parse.add_argument('--loss_params', default={'ce': ce_params,
                                             'dice': dice_params,
                                             'focal': focal_params,
                                             'lovasz': lovasz_params})

parse.add_argument('--model', default='dlv3plus', choices=['unet', 'unet_ae', 'dlv3plus', 'pspnet'], type=str)
parse.add_argument('--loss', default='ce', choices=['ce', 'dice', 'focal', 'lovasz'], type=str)
parse.add_argument('--lr', default=1e-2, type=float)
# parse.add_argument('--lr_decay_step', default=[30, 40], type=list)
# parse.add_argument('--lr_decay_rate', default=0.1, type=float)
parse.add_argument('--max_iters', default=90000, type=int)
parse.add_argument('--epochs', default=None)
parse.add_argument('--batch_size', default=16 * 1, type=int)
parse.add_argument('--distributed', default=True, type=bool)
parse.add_argument('--gpuid', default='0,1,2,3', type=str)
parse.add_argument('--num_workers', default=16, type=int)
parse.add_argument('--ckpt_dir', default='./checkpoints/')
parse.add_argument('--resume', default=False, help='resume from checkpoint', type=bool)

parse.add_argument('--train_image_size', default=768, type=int)
parse.add_argument('--val_image_size', default=(2048, 1024), help='w,h')
parse.add_argument('--in_channels', default=3, type=int)

### VOC2012 path config
# parse.add_argument('--data_type', default='voc2012', choices=['voc2012', 'cityscapes', 'coco'])
# parse.add_argument('--image_root', default='/home/yhuangcc/data/VOC2012/')
# parse.add_argument('--train_list', default='/home/yhuangcc/data/VOC2012/list/train_aug.txt')
# parse.add_argument('--val_list', default='/home/yhuangcc/data/VOC2012/list/val.txt')
# parse.add_argument('--label_file', default='/home/yhuangcc/ImageSegmentation/datasets/voc/labels')


## CityScapes path config
parse.add_argument('--data_type', default='cityscapes', choices=['voc2012', 'cityscapes', 'coco'])
parse.add_argument('--image_root', default='/home/yhuangcc/data/cityscapes/')
parse.add_argument('--train_list', default='/home/yhuangcc/data/cityscapes/train.txt')
parse.add_argument('--val_list', default='/home/yhuangcc/data/cityscapes/val.txt')
parse.add_argument('--label_file', default='/home/yhuangcc/ImageSegmentation/datasets/cityscapes/labels')


def get_config():
    config, unparsed = parse.parse_known_args()
    config.ckpt_dir = os.path.join(config.ckpt_dir, f"{config.data_type}-{config.model}-mobilenet-{config.loss}")
    if not os.path.exists(config.ckpt_dir):
        os.makedirs(config.ckpt_dir)
    logger = logging.getLogger("InfoLog")
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(os.path.join(config.ckpt_dir, 'log.txt'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)
    return config
