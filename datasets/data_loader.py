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


class CityScapesDataset(Dataset):

    def __init__(self, root, list_file, label_file, transform=None):

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

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        names = self.img_names[idx].strip().split(' ')
        img = Image.open(os.path.join(self.root, names[0][1:])).convert('RGB')
        mask = Image.open(os.path.join(self.root, names[1][1:]))
        mask = np.array(mask, dtype=np.uint8)

        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        mask = Image.fromarray(mask)

        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask.squeeze().long()


class CocoDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


def get_loader(config):
    train_size = config.train_image_size
    val_size=config.val_image_size
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    training_transforms = st.Compose([st.RandomHorizontallyFlip(),
                                      st.RandomGaussianBlur(),
                                      st.RandomRotate(degree=5),
                                      st.ColorJitter(*[0.1, 0.1, 0.1, 0.1]),
                                      st.RandomSizedCrop(size=train_size),
                                      st.ToTensor(),
                                      st.Normalize(mean, std)
                                      ])
    val_transforms = st.Compose([st.FreeScale(val_size),
                                 st.ToTensor(),
                                 st.Normalize(mean, std)
                                 ])

    if config.data_type == 'voc2012':
        DATASET = ImgSegDataset
    elif config.data_type == 'cityscapes':
        DATASET = CityScapesDataset
    elif config.data_type == 'coco':
        DATASET = CocoDataset
    else:
        raise NotImplementedError

    training_dataset = DATASET(root=config.image_root,
                               list_file=config.train_list,
                               label_file=config.label_file,
                               transform=training_transforms)
    val_dataset = DATASET(root=config.image_root,
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
