# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils import seg_transforms as st


class ImgSegDataset(Dataset):
    def __init__(self, root, list_file, label_file, transform=None):
        '''
        :param root: contains original image and mask pathes
        :param list_file: train.txt or [train.txt, val.txt] for image names
        :param label_file: class name
        :param transform:
        '''

        self.root = root
        labels = np.loadtxt(label_file, dtype=np.object, delimiter=',')
        self.labels_array = labels[:, :3].astype(int)
        self.labels_name = labels[:, 3].tolist()
        self.n_classes = len(self.labels_name)
        self.transform = transform

        if isinstance(list_file, list):
            temp_file = "/tmp/listfile.txt"
            os.system(f"cat {' '.join(list_file)} > {temp_file}")
            list_file = temp_file

        with open(list_file) as f:
            self.img_names = f.readlines()

    def __getitem__(self, idx):
        names = self.img_names[idx].strip().split(' ')
        img = Image.open(os.path.join(self.root, names[0][1:])).convert('RGB')
        mask = Image.open(os.path.join(self.root, names[1][1:]))

        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask.squeeze().long()

    def __len__(self):
        return len(self.img_names)


def get_loader(config):
    size = config.image_size
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    training_transforms = st.Compose([st.RandomHorizontallyFlip(),
                                      st.RandomGaussianBlur(),
                                      st.RandomRotate(degree=5),
                                      st.ColorJitter(*[0.1, 0.1, 0.1, 0.1]),
                                      st.RandomSizedCrop(size=size),
                                      st.ToTensor(),
                                      st.Normalize(mean, std)
                                      ])
    val_transforms = st.Compose([st.FreeScale(size),
                                 st.ToTensor(),
                                 st.Normalize(mean, std)
                                 ])

    training_dataset = ImgSegDataset(root=config.image_root,
                                     list_file=config.train_list,
                                     label_file=config.label_file,
                                     transform=training_transforms)
    val_dataset = ImgSegDataset(root=config.image_root,
                                list_file=config.val_list,
                                label_file=config.label_file,
                                transform=val_transforms)

    training_loader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True,
                                 num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, pin_memory=True)

    return {'train': training_dataset, 'val': val_dataset}, \
           {'train': training_loader, 'val': val_loader}


if __name__ == '__main__':
    training_transforms = st.Compose([st.RandomHorizontallyFlip(),
                                      st.ColorJitter(*[0.1, 0.1, 0.1, 0.1]),
                                      st.RandomSizedCrop(size=513),
                                      st.RandomGaussianBlur(),
                                      st.RandomRotate(degree=5),
                                      st.ToTensor(),
                                      st.Normalize(mean=(0.485, 0.456, 0.406),
                                                   std=(0.229, 0.224, 0.225))
                                      ])
    test_transforms = st.Compose([st.ToTensor(),
                                  st.Normalize(mean=(0.485, 0.456, 0.406),
                                               std=(0.229, 0.224, 0.225))
                                  ])
    VocDataset = ImgSegDataset(root='/home/yhuangcc/data/VOC2012/',
                               list_file='/home/yhuangcc/data/VOC2012/list/train_aug.txt',
                               label_file='/home/yhuangcc/ImageSegmentation/datasets/voc/labels',
                               transform=test_transforms)
    img, mask = VocDataset[0]
    print(img.size())
    print(mask.size())
