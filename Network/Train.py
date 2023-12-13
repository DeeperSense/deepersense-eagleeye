import pandas as pd
import argparse
import builtins
import math
import os
import random
import shutil
import time
import pandas as pd
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from datetime import datetime
import numpy as np
# import wandb
def train(train_loader, model, Cost_function, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mean_ps = AverageMeter('Mean-P', ':6.2f')
    mean_ns = AverageMeter('Mean-N', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, mean_ps, mean_ns],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images_q, images_k, indexes, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images_q = images_q.cuda(args.gpu, non_blocking=True)
            images_k = images_k.cuda(args.gpu, non_blocking=True)
            #indexes = indexes.cuda(args.gpu, non_blocking=True)

        embed_q, embed_k = model(im_q=images_q, im_k=images_k)#  , delta=delta

        loss, mean_p, mean_n = Cost_function(embed_q, embed_k)
        wandb.log({"Training Loss": loss})
        losses.update(loss.item(), images_q.size(0))
        mean_ps.update(mean_p, images_q.size(0))
        mean_ns.update(mean_n, images_q.size(0))

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        del loss
        del embed_q
        del embed_k


# save all the attention map
def scan(loader, model, args):

    model.eval()

    with torch.no_grad():
        for i, (images_q, images_k, _, indexes,) in enumerate(loader):
            if args.gpu is not None:
                images_q = images_q.cuda(args.gpu, non_blocking=True)
                images_k = images_k.cuda(args.gpu, non_blocking=True)
                indexes = indexes.cuda(args.gpu, non_blocking=True)

            # compute output
            embed_q, embed_k = model(im_q=images_q, im_k=images_k,indexes=indexes ) #,delta=delta,

def validate_patches(images_1, images_2, model, args):

    model_FLC = model.FLC_net
    model_FLS = model.FLS_net
    model_FLC.eval()
    model_FLS.eval()
    FLC_features = np.zeros([len(images_1), args.dim])
    FLC_labels1 = np.zeros([len(images_1, )])
    FLS_features = np.zeros([len(images_1, ), args.dim])

    with torch.no_grad():
        if args.gpu is not None:
            images_1 = images_2.cuda(args.gpu, non_blocking=True)
            images_2 = images_2.cuda(args.gpu, non_blocking=True)
        FLS_embed = model_FLS(x=images_1)  # delta
        FLS_features = FLS_embed.detach().cpu().numpy()
        end = time.time()
        FLC_embed = model_FLC(images_2)
        FLC_features = FLC_embed.cpu().numpy()
        FLC_labels =  [i for i in range(len(FLC_labels1))]
        [top1,similarity] = accuracy(FLC_features, FLS_features, FLC_labels)



    return top1,similarity

def validate(val_FLC_loader, val_FLS_loader, model, args):


    # switch to evaluate mode
    model_FLC = model.FLC_net
    model_FLS = model.FLS_net
    model_FLC.eval()
    model_FLS.eval()
    print('model validate on cuda', args.gpu)

    FLC_features = np.zeros([len(val_FLC_loader.dataset), args.dim])
    FLC_labels = np.zeros([len(val_FLC_loader.dataset)])
    FLS_features = np.zeros([len(val_FLS_loader.dataset), args.dim])

    with torch.no_grad():
        for i, (images, indexes ) in enumerate(val_FLS_loader):#atten
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                indexes = indexes.cuda(args.gpu, non_blocking=True)

            FLS_embed = model_FLS(x=images, indexes=indexes)  # delta
            FLS_features[indexes.cpu().numpy().astype(int), :] = FLS_embed.detach().cpu().numpy()
        end = time.time()

        # FLC features
        for i, (images, indexes, labels) in enumerate(val_FLC_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                indexes = indexes.cuda(args.gpu, non_blocking=True)
                labels = labels.cuda(args.gpu, non_blocking=True)

            # compute output
            FLC_embed = model_FLC(images)

            FLC_features[indexes.cpu().numpy(), :] = FLC_embed.cpu().numpy()
            FLC_labels[indexes.cpu().numpy()] = labels.cpu().numpy()

        [top1,similarity] = accuracy(FLC_features, FLS_features, FLC_labels.astype(int))

    if args.evaluate:
        np.save(os.path.join(args.save_path, 'grd_global_descriptor.npy'), FLC_features)
        np.save('sat_global_descriptor.npy', FLS_features)

    return top1,similarity


def save_checkpoint(state, is_best, filename='checkpoint.pth', args=None):
    torch.save(state, os.path.join(args.save_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(args.save_path, filename), os.path.join(args.save_path, 'model_best.pth'))


class AverageMeter(object):


    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


PD = pd.DataFrame()


def accuracy(FLC_features,FLS_features, FLC_labels, topk=[1, 5, 10]):#

    ts = time.time()
    N = FLC_features.shape[0]
    M = FLS_features.shape[0]
    topk.append(M // 100)
    results = np.zeros([len(topk)])
    #if N < 80000:
    FLC_features_norm = np.sqrt(np.sum(FLC_features ** 2, axis=1, keepdims=True))
    FLS_features_norm = np.sqrt(np.sum(FLS_features ** 2, axis=1, keepdims=True))
    similarity = np.matmul(FLC_features / FLC_features_norm,
                               (FLS_features / FLS_features_norm).transpose())

    for i in range(N):
        ranking = np.sum((similarity[i, :] > similarity[i, FLC_labels[i]]) * 1.)

        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.

    results = results / FLC_features.shape[0] * 100.

    print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}, time:{}'.format(results[0], results[1], results[2],
                                                                            results[-1], time.time() - ts))
    return [results[0],similarity]