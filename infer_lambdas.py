'''
CUDA_VISIBLE_DEVICES=0 python3 infer_lambdas.py  --target_domain shot_noise --bs 1 --severity 5
暴力搜索（grid searching）出每张需要的最佳lambda
'''

import argparse
import random
import sys
import time
import torch
import torch.nn.functional as F
import torchvision.models as tmodels
from utils.lerp import lerp_multi
from utils.model_path import get_model_path
from tqdm import tqdm
import numpy as np
from robustbench.data import load_imagenetc
from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreatModel

parser = argparse.ArgumentParser()
parser.add_argument('--target_domain', type=str, default='')
parser.add_argument('--severity', type=int, default=-1)
parser.add_argument('--bs', type=int, default=-1)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_lambda', type=int, default=6)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
torch.cuda.manual_seed_all(args.seed)  # 为当前GPU设置随机种子
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

args.source_domains = ['gaussian_noise', 'defocus_blur', 'snow', 'jpeg_compression']

dataloader = load_imagenetc(args.bs, args.severity, '/home/yxue/datasets', False, [args.target_domain], prepr='Res256Crop224')

param_ls = get_model_path(args)
print(len(param_ls))

lambdas = torch.linspace(0, 1, steps=args.num_lambda)
print(lambdas)
model_ls = []
for lam1 in lambdas:
    others3 = [x for x in lambdas if x + lam1 <= 1]
    for lam2 in others3:
        others2 = [x for x in lambdas if lam1 + lam2 + x <= 1]
        for lam3 in others2:
            others1 = [x for x in lambdas if lam1 + lam2 + lam3 + x == 1]
            for lam4 in others1:
                weights = torch.tensor([lam1, lam2, lam3, lam4])
                param_f = lerp_multi(param_ls, weights)
                model = load_model('Standard_R50', "./ckpt", 'imagenet', ThreatModel.corruptions)
                model.load_state_dict(param_f)
                model_ls.append(model)
print(len(model_ls))

for batch_idx, (data, label, paths) in enumerate(dataloader):  # whole test loader
    f = open(f'./暴搜结果/{args.target_domain}{args.severity}_{args.source_domains}_bs{args.bs}_seed{args.seed}_{args.num_lambda}lambdas.txt', 'a')
    f2 = open(f'./暴搜结果/{args.target_domain}{args.severity}_{args.source_domains}_bs{args.bs}_seed{args.seed}_{args.num_lambda}lambdas_loss.txt', 'a')
    s_time = time.time()
    data, label = data.cuda(), label.cuda()
    test_acc_interp_naive, test_loss_interp_naive = [], []
    with torch.no_grad():
        for model in model_ls:
            model.cuda()
            model.eval()
            output = model(data)
            model.to('cpu')
            output = F.log_softmax(output, dim=1)
            test_loss = F.nll_loss(output, label, reduction='mean').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(label.view_as(pred)).sum().item()
            acc = 100. * correct / output.size(0)
            test_acc_interp_naive.append(acc)
            test_loss_interp_naive.append(test_loss)
    f.write(f"{test_acc_interp_naive}\n")
    f2.write(f"{test_loss_interp_naive}\n")
    e_time = time.time()
    print(f'batch{batch_idx} time: {e_time-s_time}')
    f.close()
    f2.close()