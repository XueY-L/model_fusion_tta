'''
CUDA_VISIBLE_DEVICES=0 python -u train.py --cfg train.yaml 
'''
import logging
import random
import numpy as np
import torch
import torch.optim as optim
import torchvision.models as tmodels
from robustbench.data import load_imagenetc
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from conf import cfg, load_cfg_fom_args
from single_domain_trainer import Trainer

load_cfg_fom_args('Imagenet-C training')
# configure model
base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions)

param = torch.load('/home/yxue/TTA-try/checkpoint/ckpt_[\'jpeg_compression\']_[5].pt', map_location='cpu')['model']
base_model.load_state_dict(param)

optimizer = torch.optim.SGD(
    # filter(lambda p: p.requires_grad, nets.model.parameters()),
    base_model.parameters(),
    lr=cfg.OPTIM.LR, 
    momentum=cfg.OPTIM.MOMENTUM, 
    weight_decay=cfg.OPTIM.WD,
)

trainer = Trainer(
    base_model,
    optimizer,
    epochs=50,
    _C=cfg, 
)

trainer.train()
# trainer.evaluate()