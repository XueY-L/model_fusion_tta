'''
CUDA_VISIBLE_DEVICES=1 python -u train.py --cfg train_cifar100c.yaml 
'''
import sys
sys.path.append('/home/yxue/model_fusion_tta')
import logging
import random
import numpy as np
import torch
import torchvision.models as tmodels
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from conf_cifar100c import cfg, load_cfg_fom_args_cifar100c
from single_domain_trainer import Trainer

load_cfg_fom_args_cifar100c('Cifar100C training')
# configure model
base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR, cfg.CORRUPTION.DATASET, ThreatModel.corruptions)
print(base_model)

# param = torch.load('checkpoint/ckpt_[\'jpeg_compression\']_[5].pt', map_location='cpu')['model']
# base_model.load_state_dict(param)

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
    epochs=20,
    _C=cfg, 
)

# trainer.train()
# trainer.evaluate(flag=False)