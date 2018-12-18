# -*- coding: utf-8 -*-
import json
import os
import logging
import torch
from torch import optim, nn
import torch.nn.functional as F
from datasets import get_loader
from loss import LossSelector
from models import ModelSelector
from utils import ScoreMeter, AveMeter, Timer, patch_replication_callback
from utils.visualization import decode_mask_seq
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

logger = logging.getLogger('InfoLog')


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.ckpt_dir = config.ckpt_dir
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.save_config(config)
        self.timer = Timer()

        self.writer = SummaryWriter(log_dir=config.ckpt_dir)

        self.lr = config.lr
        self.datasets, self.loaders = get_loader(config)
        self.max_iters = config.max_iters
        if self.max_iters is not None:
            self.epochs = self.max_iters // len(self.loaders['train'])
        else:
            self.epochs = config.epochs
        self.start_epoch = 0
        self.num_classes = self.datasets['train'].n_classes

        self.scores = ScoreMeter(self.num_classes)

        self.model = ModelSelector[config.model](in_channels=config.in_channels,
                                                 num_classes=self.num_classes,
                                                 **config.model_params[config.model])

        if config.distributed:
            self.model = nn.DataParallel(self.model)
            patch_replication_callback(self.model)

        self.model = self.model.cuda()

        self.criterion = LossSelector[config.loss](**config.loss_params[config.loss])
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=4e-5)
        self.lr_decay = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_iters)

        self.best_miou = float('-inf')

        if config.resume:
            logger.info('***Resume from checkpoint***')
            state = torch.load(os.path.join(self.ckpt_dir, 'ckpt.pt'))
            self.model.load_state_dict(state['model'])
            self.start_epoch = state['epoch']
            self.best_miou = state['best_miou']
            self.optimizer.load_state_dict(state['optim'])
            self.lr_decay.load_state_dict(state['lr_decay'])
            self.lr_decay.last_epoch = self.start_epoch

    def train_and_val(self):
        for epoch in range(self.start_epoch, self.epochs):
            logger.info(f"Epoch :{epoch}")
            self.train(epoch)
            logger.info(f"val starts...")
            val_miou, val_cls_iou = self.val(epoch)
            if val_miou > self.best_miou:
                logger.info('------')
                self.best_miou = val_miou
                self.save({'net': self.model.state_dict(),
                           'best_miou': val_miou,
                           'epoch': epoch,
                           'optim': self.optimizer.state_dict(),
                           'lr_decay': self.lr_decay.state_dict()})
        logger.info(f"per class iou: {val_cls_iou}")

        self.writer.close()

    def train(self, epoch):
        losses = AveMeter()
        self.scores.reset()

        self.model.train()
        for i, (imgs, targets) in enumerate(self.loaders['train']):
            self.lr_decay.step()

            imgs = imgs.cuda()
            targets = targets.cuda()

            outs = self.model(imgs)
            if not isinstance(outs, tuple):
                loss = self.criterion(outs, targets)
                self.scores.update(targets.cpu().data.numpy(),
                                   outs.argmax(dim=1).cpu().data.numpy())

            elif len(outs) == 2:
                # For pspnet outputs
                try:
                    loss = self.criterion(outs, targets)
                    self.scores.update(targets.cpu().data.numpy(),
                                       outs[1].argmax(dim=1).cpu().data.numpy())
                except:
                    assert self.config.loss in ['ce', 'focal'], 'For pspnet, only ce and focal can support'

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), imgs.size()[0])
            scores, _ = self.scores.get_scores()

            if i % 200 == 0 or i == len(self.loaders['train']) - 1:
                logger.info(f"Train: [{i}/{len(self.loaders['train'])}] | "
                            f"Time: {self.timer.timeSince()} | "
                            f"loss: {losses.avg:.4f} | "
                            f"oa:{scores['oa']:.4f} | "
                            f"ma: {scores['ma']:.4f} | "
                            f"fa: {scores['fa']:.4f} | "
                            f"miou: {scores['miou']:.4f}")

        self.writer.add_scalar('train/loss', losses.avg, epoch)
        self.writer.add_scalar('train/mIoU', scores['miou'], epoch)
        self.writer.add_scalar('train/Aacc', scores['oa'], epoch)
        self.writer.add_scalar('train/Acc_class', scores['ma'], epoch)
        self.writer.add_scalar('train/Acc_freq', scores['fa'], epoch)

    def val(self, epoch):
        losses = AveMeter()
        self.scores.reset()

        self.model.eval()
        with torch.no_grad():
            for i, (imgs, targets) in enumerate(self.loaders['val']):
                imgs = imgs.cuda()
                targets = targets.cuda()

                outs = self.model(imgs)

                loss = self.criterion(outs, targets)

                self.scores.update(targets.cpu().data.numpy(),
                                   outs.argmax(dim=1).cpu().data.numpy())
                losses.update(loss.item(), imgs.size()[0])

            scores, cls_iou = self.scores.get_scores()

            logger.info(f"Val: [{i}/{len(self.loaders['val'])}] | "
                        f"Time: {self.timer.timeSince()} | "
                        f"loss: {losses.avg:.4f} | "
                        f"oa:{scores['oa']:.4f} | "
                        f"ma: {scores['ma']:.4f} | "
                        f"fa: {scores['fa']:.4f} | "
                        f"miou: {scores['miou']:.4f}")

        self.writer.add_scalar('val/loss', losses.avg, epoch)
        self.writer.add_scalar('val/mIoU', scores['miou'], epoch)
        self.writer.add_scalar('val/Acc', scores['oa'], epoch)
        self.writer.add_scalar('val/Acc_class', scores['ma'], epoch)
        self.writer.add_scalar('val/Acc_freq', scores['fa'], epoch)

        if epoch % 10 == 0:
            self.summary_imgs(imgs, targets, outs, epoch)

        return scores['miou'], cls_iou

    def save(self, state):
        torch.save(state, os.path.join(self.ckpt_dir, 'ckpt.pt'))
        logger.info('***Saving model***')

    def save_config(self, config):
        with open(os.path.join(self.ckpt_dir, 'config.json'), 'w+') as f:
            f.write(json.dumps(config.__dict__, indent=4))
        f.close()

    def summary_imgs(self, imgs, targets, outputs, epoch):
        grid_imgs = make_grid(imgs[:3].clone().cpu().data, nrow=3, normalize=True)
        self.writer.add_image('Image', grid_imgs, epoch)
        grid_imgs = make_grid(decode_mask_seq(outputs[:3].argmax(1).cpu().data.numpy(),
                                              self.datasets['val'].labels_array),
                              nrow=3, normalize=False, range=(0, 255))
        self.writer.add_image('Predicted mask', grid_imgs, epoch)
        grid_imgs = make_grid(decode_mask_seq(targets[:3].cpu().data.numpy(),
                                              self.datasets['val'].labels_array),
                              nrow=3, normalize=False, range=(0, 255))
        self.writer.add_image('GT mask', grid_imgs, epoch)
