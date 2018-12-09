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
        labels = np.loadtxt(label_file, dtype=np.object)
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

    @staticmethod
    def encode_mask(mask, labels):
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(labels):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    @staticmethod
    def decode_mask(label_mask, labels, plot=False):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        if isinstance(label_mask, torch.Tensor):
            label_mask = label_mask.cpu().numpy()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, len(labels)):
            r[label_mask == ll] = labels[ll, 0]
            g[label_mask == ll] = labels[ll, 1]
            b[label_mask == ll] = labels[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

    @staticmethod
    def decode_mask_seq(label_masks, labels):
        rgb_masks = []
        for label_mask in label_masks:
            if isinstance(label_mask, torch.Tensor):
                label_mask = label_mask.cpu().numpy()
            r = label_mask.copy()
            g = label_mask.copy()
            b = label_mask.copy()
            for ll in range(0, len(labels)):
                r[label_mask == ll] = labels[ll, 0]
                g[label_mask == ll] = labels[ll, 1]
                b[label_mask == ll] = labels[ll, 2]
            rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
            rgb[:, :, 0] = r / 255.0
            rgb[:, :, 1] = g / 255.0
            rgb[:, :, 2] = b / 255.0
            rgb_masks.append(rgb)
        return torch.from_numpy(np.array(rgb_masks).transpose(0, 3, 1, 2))


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
