'''
python3 /home/yxue/git-re-basin-pytorch/scripts/batchwise/5source/暴搜形成分布/找到topk最优权重.py --num_lambda 11
'''

import sys
sys.path.append('/home/yxue/git-re-basin-pytorch')
import heapq
import random
import torch
import torch.nn.functional as F
import torchvision.models as tmodels
import numpy as np
from dataloaders import DomainNetLoader
from utils.utils import lerp_multi, load_model, AverageMeter, get_lambdas
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--target_domain', type=str, default='')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_lambda', type=int, default=6)
args = parser.parse_args()

# Fix seeds for reproducibility
# These five lines control all the major sources of randomness.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
torch.cuda.manual_seed_all(args.seed)  # 为当前GPU设置随机种子
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# path = '/home/yxue/git-re-basin-pytorch/scripts/batchwise/5source/param_fusion_acc_loss/LODO-test集合暴搜（bs=32, 11lambdas）/C_seed42_11lambdas_LODO.txt'

path = '/home/yxue/git-re-basin-pytorch/scripts/batchwise/5source/param_fusion_acc_loss/LODO-test集合暴搜（bs=1，11lambdas）/C-train_[\'infograph\', \'painting\', \'quickdraw\', \'real\', \'sketch\']_seed42_11lambdas_LODO_lr0.001.txt'


f = open(path, 'r')
lines = f.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].strip()[1:-1].split(', ')
    lines[i] = [float(x) for x in lines[i]]

counts = [0] * len(lines[0])
for i in range(len(lines)):  # len(lines)
    amax = np.where(lines[i] == np.max(100.0))  # acc最大的
    # amax = np.where(lines[i] == np.min(lines[i]))  # CE loss最小的
    for idx in amax[0]:
        counts[idx] += 1
# print(counts)

k = 15
topk = heapq.nlargest(k, range(len(counts)), counts.__getitem__)  # nsmallest
print(topk, [counts[index] for index in topk])

weight_ls = get_lambdas(div=args.num_lambda)
print(len(weight_ls), len(counts))

res = []
for idx in topk:
    res.append(weight_ls[idx].data)
print(res)