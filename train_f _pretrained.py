import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from datetime import datetime
#import matplotlib.pyplot as plt
import numpy as np
from dataset.VIGOR import VIGOR
from dataset.CVUSA import CVUSA
from dataset.CVACT import CVACT
from model.TransGeo import TransGeo
from criterion.soft_triplet import SoftTripletBiLoss
from dataset.global_sampler import DistributedMiningSampler,DistributedMiningSamplerVigor
from criterion.sam import SAM
from ptflops import get_model_complexity_info



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')#256
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=r"D:\fabian\PycharmProjects\transgeo1\pretrained_model_CVUSA\model_best.pth", type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_path', default=r'D:\fabian\PycharmProjects\transgeo1\results', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default= 0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

# moco specific configs:
parser.add_argument('--dim', default=1000, type=int,
                    help='feature dimension (default: 128)')

# options for moco v2
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

parser.add_argument('--cross', action='store_true',
                    help='use cross area')

parser.add_argument('--dataset', default='cvusa', type=str,
                    help='vigor, cvusa, cvact')
parser.add_argument('--op', default='sgd', type=str,
                    help='sgd, adam, adamw')

parser.add_argument('--share', action='store_true',
                    help='share fc')

parser.add_argument('--mining', action='store_true',
                    help='mining')
parser.add_argument('--asam', action='store_true',
                    help='asam')

parser.add_argument('--rho', default=0.05, type=float,
                    help='rho for sam')
parser.add_argument('--sat_res', default=0, type=int,
                    help='resolution for satellite')

parser.add_argument('--crop', action='store_true',
                    help='nonuniform crop')

parser.add_argument('--fov', default=0, type=int,
                    help='Fov')


best_acc1 = 0


def compute_complexity(model, args):
    size_sat = [224, 224]  # [512, 512]
    size_sat_default = [224, 224]  # [512, 512]
    size_grd = [224, 224]  # [224, 1232]
    if args.sat_res != 0:
        size_sat = [args.sat_res, args.sat_res]
    if args.fov != 0:
        size_grd[1] = int(args.fov /360. * size_grd[1])
    with torch.cuda.device(0):
        macs_1, params_1 = get_model_complexity_info(model.query_net, (3, size_grd[0], size_grd[1]), as_strings=False,
                                                 print_per_layer_stat=True, verbose=True)
        macs_2, params_2 = get_model_complexity_info(model.reference_net, (3, size_sat[0] , size_sat[1] ),
                                                     as_strings=False,
                                                     print_per_layer_stat=True, verbose=True)

        print('flops:', (macs_1+macs_2)/1e9, 'params:', (params_1+params_2)/1e6)



def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1

    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node
    print("Use GPU: {} for training".format(args.gpu))
    print("=> creating model '{}'")

    model = TransGeo(args=args)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    #compute_complexity(model, args)  # uncomment to see detailed computation cost

    criterion = SoftTripletBiLoss().cuda(args.gpu)

    if args.op == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.op == 'adam':
        optimizer = torch.optim.Adam(parameters, args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    elif args.op == 'adamw':
        optimizer = torch.optim.AdamW(parameters, args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    elif args.op == 'sam':
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(parameters, base_optimizer,  lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False, rho=args.rho, adaptive=args.asam)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            if not args.crop:
                args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']

            for key in list(checkpoint['state_dict'].keys()):
               checkpoint['state_dict'][key.replace("module.","")]=checkpoint['state_dict'].pop(key)
            #checkpoint['state_dict']['query_net.cls_token']=checkpoint['state_dict'].pop('module.query_net.cls_token')
            """for i in checkpoint['state_dict'].keys():
               print(i)"""
            checkpoint['state_dict']['query_net.pos_embed']=checkpoint['state_dict']['query_net.pos_embed'][:,35:233,:]
            checkpoint['state_dict']['reference_net.pos_embed'] = checkpoint['state_dict']['reference_net.pos_embed'][:, 102:300,:]
            model.load_state_dict(checkpoint['state_dict'])
            if args.op == 'sam' and args.dataset != 'cvact':    # Loading the optimizer status gives better result on CVUSA, but not on CVACT.
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if not args.multiprocessing_distributed or args.gpu == 0:
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
            os.mkdir(os.path.join(args.save_path, 'attention'))
            os.mkdir(os.path.join(args.save_path, 'attention','train'))
            os.mkdir(os.path.join(args.save_path, 'attention','val'))

    dataset = CVUSA


    train_dataset = dataset(mode='train', print_bool=True,args=args)
    val_scan_dataset = dataset(mode='scan_val',  args=args)
    val_query_dataset = dataset(mode='test_query',  args=args)
    val_reference_dataset = dataset(mode='test_reference',  args=args)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    train_scan_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,sampler=train_sampler, drop_last=False)#sampler=torch.utils.data.distributed.DistributedSampler(train_dataset)

    val_scan_loader = torch.utils.data.DataLoader(
        val_scan_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        drop_last=False)#sampler=torch.utils.data.distributed.DistributedSampler(val_scan_dataset)

    val_query_loader = torch.utils.data.DataLoader(
        val_query_dataset,batch_size=8, shuffle=False,
        num_workers=args.workers, pin_memory=True) # 512, 64
    val_reference_loader = torch.utils.data.DataLoader(
        val_reference_dataset, batch_size=8, shuffle=False,
        num_workers=args.workers, pin_memory=True) # 80, 128

    if args.evaluate:
        if not args.multiprocessing_distributed or args.gpu == 0:
            validate(val_query_loader, val_reference_loader, model, args)
        return

    for epoch in range(args.start_epoch, args.epochs):

        print('start epoch:{}, date:{}'.format(epoch, datetime.now()))
        if args.distributed:
            train_sampler.set_epoch(epoch)
            if args.mining:
                train_sampler.update_epoch()
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, train_sampler)

        # evaluate on validation set
        if not args.multiprocessing_distributed or args.gpu == 0:
            acc1 = validate(val_query_loader, val_reference_loader, model, args)
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
        #torch.distributed.barrier()

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename='checkpoint.pth.tar', args=args) # 'checkpoint_{:04d}.pth.tar'.format(epoch)

    if not args.crop:
        model.reference_net.save = os.path.join(args.save_path, 'attention', 'train')
        scan(train_scan_loader, model, args)
        model.reference_net.save = os.path.join(args.save_path, 'attention', 'val')
        scan(val_scan_loader, model, args)


def train(train_loader, model, criterion, optimizer, epoch, args, train_sampler=None):
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
    for i, (images_q, images_k, indexes, _, delta, atten) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images_q = images_q.cuda(args.gpu, non_blocking=True)
            images_k = images_k.cuda(args.gpu, non_blocking=True)
            indexes = indexes.cuda(args.gpu, non_blocking=True)

        # compute output
        if args.crop:
            embed_q, embed_k = model(im_q=images_q, im_k=images_k, delta=delta, atten=atten)
        else:
            embed_q, embed_k = model(im_q =images_q, im_k=images_k, delta=delta)

        loss, mean_p, mean_n = criterion(embed_q, embed_k)

        if args.mining:
            train_sampler.update(concat_all_gather(embed_k).detach().cpu().numpy(),concat_all_gather(embed_q).detach().cpu().numpy(),concat_all_gather(indexes).detach().cpu().numpy())
        losses.update(loss.item(), images_q.size(0))
        mean_ps.update(mean_p, images_q.size(0))
        mean_ns.update(mean_n, images_q.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.op != 'sam':
            optimizer.step()
        else:
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass, only for ASAM
            if args.crop:
                embed_q, embed_k = model(im_q=images_q, im_k=images_k, delta=delta, atten=atten)
            else:
                embed_q, embed_k = model(im_q=images_q, im_k=images_k, delta=delta)

            loss, mean_p, mean_n = criterion(embed_q, embed_k)
            loss.backward()
            optimizer.second_step(zero_grad=True)

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
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time],
        prefix="Scan:")

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images_q, images_k, _, indexes , delta, _) in enumerate(loader):

            # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                images_q = images_q.cuda(args.gpu, non_blocking=True)
                images_k = images_k.cuda(args.gpu, non_blocking=True)
                indexes = indexes.cuda(args.gpu, non_blocking=True)

            # compute output
            embed_q, embed_k = model(im_q =images_q, im_k=images_k, delta=delta, indexes=indexes)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)


# query features and reference features should be computed separately without correspondence label
def validate(val_query_loader, val_reference_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    progress_q = ProgressMeter(
        len(val_query_loader),
        [batch_time],
        prefix='Test_query: ')
    progress_k = ProgressMeter(
        len(val_reference_loader),
        [batch_time],
        prefix='Test_reference: ')

    # switch to evaluate mode
    model_query = model.query_net
    model_reference = model.reference_net
    model_query.eval()
    model_reference.eval()
    print('model validate on cuda', args.gpu)

    query_features = np.zeros([len(val_query_loader.dataset), args.dim])
    query_labels = np.zeros([len(val_query_loader.dataset)])
    reference_features = np.zeros([len(val_reference_loader.dataset), args.dim])

    with torch.no_grad():
        end = time.time()
        # reference features
        for i, (images, indexes, atten) in enumerate(val_reference_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                indexes = indexes.cuda(args.gpu, non_blocking=True)

            # compute output
            if args.crop:
                reference_embed = model_reference(x=images, atten=atten)
            else:
                reference_embed = model_reference(x=images, indexes=indexes)  # delta

            reference_features[indexes.cpu().numpy().astype(int), :] = reference_embed.detach().cpu().numpy()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress_k.display(i)

        end = time.time()

        # query features
        for i, (images, indexes, labels) in enumerate(val_query_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                indexes = indexes.cuda(args.gpu, non_blocking=True)
                labels = labels.cuda(args.gpu, non_blocking=True)

            # compute output
            query_embed = model_query(images)

            query_features[indexes.cpu().numpy(), :] = query_embed.cpu().numpy()
            query_labels[indexes.cpu().numpy()] = labels.cpu().numpy()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress_q.display(i)

        [top1, top5] = accuracy(query_features, reference_features, query_labels.astype(int))

    if args.evaluate:
        np.save(os.path.join(args.save_path, 'grd_global_descriptor.npy'), query_features)
        np.save('sat_global_descriptor.npy', reference_features)

    return top1


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', args=None):
    torch.save(state, os.path.join(args.save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(args.save_path,filename), os.path.join(args.save_path,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def accuracy(query_features, reference_features, query_labels, topk=[1,5,10]):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    ts = time.time()
    N = query_features.shape[0]
    M = reference_features.shape[0]
    topk.append(M//100)
    results = np.zeros([len(topk)])
    # for CVUSA, CVACT
    if N < 80000:
        query_features_norm = np.sqrt(np.sum(query_features**2, axis=1, keepdims=True))
        reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
        similarity = np.matmul(query_features/query_features_norm, (reference_features/reference_features_norm).transpose())

        for i in range(N):
            ranking = np.sum((similarity[i,:]>similarity[i,query_labels[i]])*1.)

            for j, k in enumerate(topk):
                if ranking < k:
                    results[j] += 1.
    

    results = results/ query_features.shape[0] * 100.
    print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}, time:{}'.format(results[0], results[1], results[2], results[-1], time.time() - ts))
    return results[:2]

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == '__main__':
    main()
