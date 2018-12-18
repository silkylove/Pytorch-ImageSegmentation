# -*- coding: utf-8 -*-
import os
import re

root = '/home/yhuangcc/data/cityscapes/'
for phase in ['train', 'val', 'test']:
    with open(root + f'{phase}.txt', 'w+') as f:
        for sub_path, sub_dir, files in os.walk(os.path.join(root, f'leftImg8bit/{phase}/')):
            if len(files) != 0:
                for file in files:
                    name1 = f'/leftImg8bit/{phase}/' + sub_path.split('/')[-1] + '/' + file
                    name2 = f'/gtFine/{phase}/' + sub_path.split('/')[-1] + '/' + \
                            re.sub('leftImg8bit', 'gtFine_labelIds', file)
                    f.write(name1 + ' ' + name2 + '\n')
