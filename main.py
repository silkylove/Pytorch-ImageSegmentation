# -*- coding: utf-8 -*-
import os
import logging
from trainer import Trainer
from config import get_config

logger = logging.getLogger('InfoLog')


def main(config):
    logger.info(f'***START TRAINING {config.model}***'.upper())
    trainer = Trainer(config)
    trainer.train_and_val()


if __name__ == '__main__':
    config = get_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpuid
    main(config)
