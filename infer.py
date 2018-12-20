# -*- coding: utf-8 -*-
import os
import time
import json
import torch
import numpy as np
import click
from PIL import Image
from models import ModelSelector
from torchvision import transforms
from utils.visualization import decode_mask
import matplotlib.pyplot as plt


@click.command()
@click.option('--img_path', default='../data/VOC2012/JPEGImages/2007_000032.jpg',
              prompt='Your image path:', help='image path')
@click.option('--ckpt_path', default='./checkpoints/voc2012-dlv3plus-ce/',
              prompt='Your model and config path:', help='model config path')
@click.option('--use_gpu', default=[1], prompt='use gpu or not', help='gpu')
def main(img_path, ckpt_path, use_gpu):
    class Config(object):
        def __init__(self, j):
            self.__dict__ = json.load(j)

    config_path = os.path.join(ckpt_path, 'config.json')
    ckpt_path = os.path.join(ckpt_path, 'ckpt.pt')

    config = Config(open(config_path, 'r'))
    label_file = config.label_file
    labels = np.loadtxt(label_file, dtype=np.object, delimiter=',')
    labels_array = labels[:, :3].astype(int)
    labels_name = labels[:, 3].tolist()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         (0.229, 0.224, 0.225))])
    size = config.val_image_size
    if isinstance(size, int):
        size = (int(size), int(size))
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    img = img.resize(size, Image.BILINEAR)
    input = transform(img)

    model = ModelSelector[config.model](in_channels=config.in_channels,
                                        num_classes=len(labels_name),
                                        **config.model_params[config.model])
    model = torch.nn.DataParallel(model, use_gpu)
    if use_gpu:
        device = torch.device(use_gpu[0])
        input = input.to(device)
        model = model.to(device)
        ckpt = torch.load(ckpt_path)['net']
    else:
        ckpt = torch.load(ckpt_path, 'cpu')['net']

    model.load_state_dict(ckpt, strict=False)
    model.eval()
    start = time.time()
    output = model(input.unsqueeze(0))
    print(f'Total forward time: {time.time()-start:.4f}s')
    predict_mask = decode_mask(output.argmax(1).cpu().squeeze().data.numpy(),
                               labels=labels_array)
    predict_mask = Image.fromarray((predict_mask * 255).astype(np.uint8))
    fig, ax = plt.subplots(2)
    ax[0].imshow(img.resize((w, h)))
    ax[0].axis('off')
    ax[1].imshow(predict_mask.resize((w, h)))
    ax[1].axis('off')
    plt.show()


if __name__ == '__main__':
    main()
