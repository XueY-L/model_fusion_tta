'''
Train nets.model on DomainNet
'''
import os
import torch
import torch.utils.data
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import wandb
from utils.averageMeter import AverageMeter
from robustbench.data import load_imagenetc, load_cifar100c, load_cifar10c


class Trainer:
    def __init__(
        self, 
        model, 
        optimizer,
        epochs,
        _C,
    ) -> None:
        self.model = model.cuda()
        self.optimizer = optimizer
        self.loss = nn.CrossEntropyLoss()
        self.epochs = epochs
        self._C = _C
        self.start_epoch = -1
        
        # lr decay
        # self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self._C.OPTIM.LR_DECAY, gamma=0.1)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

    def evaluate(self, phase="test", best_acc=999, epoch=-1, flag=True):
        self.model.eval()
        losses, top1 = AverageMeter(), AverageMeter()

        # 训练数据集
        if 'imagenet' in self._C.CORRUPTION.DATASET:
            self.test_loader = load_imagenetc(self._C.TEST.BATCH_SIZE, self._C.CORRUPTION.SEVERITY[0], self._C.DATA_DIR, False, self._C.CORRUPTION.TYPE, prepr='Res256Crop224')
        elif 'cifar100' == self._C.CORRUPTION.DATASET:
            self.test_loader = self.train_loader  # 训练和测试的数据增强一样
        elif 'cifar10' == self._C.CORRUPTION.DATASET:
            self.test_loader = self.train_loader  # 训练和测试的数据增强一样

        if flag:
            with torch.no_grad():
                for batch_idx, (data, label, path) in enumerate(self.test_loader):
                    if batch_idx < self.num_train_batch: continue
                    data, label = data.cuda(), label.cuda()
                    rst = self.model(data)
                    l = self.loss(rst, label)

                    losses.update(l.item(), label.size(0))

                    _, predicted = torch.max(rst.data, 1)
                    correct = predicted.eq(label.data).cpu().sum()
                    # print(rst, predicted, correct)
                    top1.update(correct*100./label.size(0), label.size(0))
            
            print('\n{phase} Loss {loss.avg:.3f}\t'
            '{phase} Acc {top1.avg:.3f}'.format(phase=phase, loss=losses, top1=top1))
        
        # Save checkpoint
        acc = top1.avg
        if acc > best_acc:
            print('Saving..')
            checkpoint = {
                "model": self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                "epoch": epoch,
                "acc": acc, 
            }
            torch.save(
                checkpoint, 
                f'checkpoint/ckpt_{self._C.CORRUPTION.DATASET}_{self._C.CORRUPTION.TYPE}_{self._C.CORRUPTION.SEVERITY}.pt'
            )
            
            best_acc = acc
    
        return top1.avg, losses.avg, best_acc

    def train_epoch(self, epoch,):
        self.model.train()
        print('\nEpoch: %d' % epoch)
        losses = AverageMeter()
        top1 = AverageMeter()
        
        for batch_idx, (data, label, paths) in enumerate(self.train_loader):
            if batch_idx == self.num_train_batch: break
            data, label = data.cuda(), label.cuda()
            rst = self.model(data)
            l = self.loss(rst, label)
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
            
            losses.update(l.item(), label.size(0))

            _, predicted = torch.max(rst.data, 1)
            correct = predicted.eq(label.data).cpu().sum()
            # print(label.size(0))
            # print(rst.softmax(-1).max(-1))
            # print(label, predicted, correct, top1.val)
            top1.update(correct*100./label.size(0), label.size(0))

            # Log per-iteration data in wandb
            self.iteration = batch_idx + epoch * self.num_train_batch
            wandb.log({"Iter Train Loss": losses.val}, step=self.iteration)
            wandb.log({"Iter Train Acc": top1.val}, step=self.iteration)

            if batch_idx % 50 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                        'Lr {lr}\t'
                        'Loss {loss.avg:.4f}\t'
                        'Acc {top1.avg:.3f}'.format(epoch, batch_idx, self.num_train_batch, 
                        lr=self.optimizer.state_dict()['param_groups'][0]['lr'], loss=losses, top1=top1))
        
        # Log per-iteration data in wandb
        wandb.log({"Epoch Train Loss": losses.avg}, step=self.iteration)
        wandb.log({"Epoch Train Acc": top1.avg}, step=self.iteration)

        return top1.avg, losses.avg

    def train(self):
        # Setup wandb
        path = os.path.join('./', 'wandb_runs')
        if not os.path.isdir(path):
            os.mkdir(path)
        wandb.init(
            project='TTA',
            name=f'{self._C.CORRUPTION.DATASET}_{self._C.CORRUPTION.TYPE}_{self._C.CORRUPTION.SEVERITY}_training',
            dir=path,
        )
        wandb.watch(self.model)
        wandb.config.update(self._C)

        # 训练数据集
        if 'imagenet' in self._C.CORRUPTION.DATASET:
            self.train_loader = load_imagenetc(self._C.TEST.BATCH_SIZE, self._C.CORRUPTION.SEVERITY[0], self._C.DATA_DIR, True, self._C.CORRUPTION.TYPE, prepr='train')
            self.num_train_batch = int(5000 * len(self._C.CORRUPTION.TYPE) // self._C.TEST.BATCH_SIZE * 0.9)
        elif 'cifar100' == self._C.CORRUPTION.DATASET:
            x_test, y_test = load_cifar100c(self._C.CORRUPTION.NUM_EX*len(self._C.CORRUPTION.TYPE), self._C.CORRUPTION.SEVERITY[0], self._C.DATA_DIR, True, self._C.CORRUPTION.TYPE)
            # print(x_test.size(), y_test.size())
            dataset = TensorDataset(x_test, y_test, torch.tensor([0]*(y_test.size(0))))
            self.train_loader = DataLoader(dataset, batch_size=self._C.TEST.BATCH_SIZE, shuffle=True)
            self.num_train_batch = int(self._C.CORRUPTION.NUM_EX * len(self._C.CORRUPTION.TYPE) // self._C.TEST.BATCH_SIZE * 0.9)
        elif self._C.CORRUPTION.DATASET == 'cifar10':
            x_test, y_test = load_cifar10c(self._C.CORRUPTION.NUM_EX*len(self._C.CORRUPTION.TYPE), self._C.CORRUPTION.SEVERITY[0], self._C.DATA_DIR, True, self._C.CORRUPTION.TYPE)
            print(x_test.size(), y_test.size())
            dataset = TensorDataset(x_test, y_test, torch.tensor([0]*(y_test.size(0))))
            self.train_loader = DataLoader(dataset, batch_size=self._C.TEST.BATCH_SIZE, shuffle=True)
            self.num_train_batch = int(self._C.CORRUPTION.NUM_EX * len(self._C.CORRUPTION.TYPE) // self._C.TEST.BATCH_SIZE * 0.9)

            
        best_acc = -1
        print(f'Number of training batches: {self.num_train_batch}')

        for epoch in range(self.start_epoch+1, self.epochs):
            train_acc, train_loss = self.train_epoch(epoch)
            self.lr_scheduler.step()
            val_acc, val_loss, best_acc = self.evaluate(phase='val', best_acc=best_acc, epoch=epoch, flag=True)
            
            wandb.log({"Iteration": self.iteration, "Epoch": epoch}, step=self.iteration)
            wandb.log({"Validation Loss": val_loss}, step=self.iteration)
            wandb.log({"Validation Accuracy": val_acc}, step=self.iteration)
