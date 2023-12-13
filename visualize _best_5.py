import argparse
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
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import numpy as np
from dataset.CVUSA import CVUSA
from model.TransGeo import TransGeo
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2
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
parser.add_argument('--resume', default=r"D:\fabian\PycharmProjects\transgeo1\results_15_1_sonar_straight\model_best.pth", type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_path', default=r'D:\fabian\PycharmProjects\transgeo1\results', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
"""parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')"""
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

            model.load_state_dict(checkpoint['state_dict'],)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    dataset = CVUSA

    train_dataset = dataset(mode='train', print_bool=True, args=args)
    val_scan_dataset = dataset(mode='scan_val', args=args)
    val_query_dataset = dataset(mode='test_query',  args=args)
    val_reference_dataset = dataset(mode='test_reference',  args=args)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    train_scan_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,sampler=train_sampler, drop_last=False)

    val_scan_loader = torch.utils.data.DataLoader(
        val_scan_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        drop_last=False)

    val_query_loader = torch.utils.data.DataLoader(
        val_query_dataset,batch_size=8, shuffle=False,
        num_workers=args.workers, pin_memory=True) # 512, 64
    val_reference_loader = torch.utils.data.DataLoader(
        val_reference_dataset, batch_size=8, shuffle=False,
        num_workers=args.workers, pin_memory=True) # 80, 128
    args.evaluate= True
    test_id= val_query_dataset.id_test_list
    best_5(val_query_loader, val_reference_loader, model, args, test_id)



def best_5(val_query_loader, val_reference_loader, model, args,test_id):

    adresses = []
    similarity = validate(val_query_loader, val_reference_loader, model, args)[1]
    if not os.path.exists(os.path.join(args.save_path, 'best 5')):
        os.mkdir(os.path.join(args.save_path, 'best 5'))
    for j in range(len(test_id)):
        first_row = pd.DataFrame(similarity[j, :], columns=['col1'])  # .sort_values(by=['col1'], ascending=False)
        first_row = first_row.sort_values(by=['col1'], ascending=False)
        first_5 = first_row.head(5)
        indexes_5 = first_5.index.tolist()
        # print(test_id[1])
        adresses = [test_id[i] for i in indexes_5]


        fig = plt.figure(figsize=(10, 3), dpi=160)
        plt.axis('off')

        img = cv2.imread(
            os.path.join(r'E:\rov_data\update4_eilat', test_id[j][1].replace("/docker/bags/", "").replace('\n', "")))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((224, 224))
        n = 0
        # for t in (0, 112):
        #      for i in (0,112) :
        #          area = (i,t, i+112, t+112)
        #          cropped_img = img.crop(area)
        #          cropped_img.save(os.path.join(r"D:\fabian\PycharmProjects\Dual\results\quarters\sonar\sonar", (str(n) + ".jpg")))
        #          #cropped_img.show()
        #          n+=1
        fig.add_subplot(2, 5, 3)
        plt.imshow(img)
        plt.axis('off')
        # img = Image.open(test_id[j][2])
        img = cv2.imread(
            os.path.join(r'E:\rov_data\update4_eilat', test_id[j][2].replace("/docker/bags/", "").replace('\n', "")))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((224, 224))
        n = 0
        # for t in (0, 112):
        #      for i in (0, 112):
        #          area = (i, t, i + 112, t + 112)
        #          cropped_img = img.crop(area)
        #          cropped_img.save(os.path.join(r"D:\fabian\PycharmProjects\Dual\results\quarters\flc\flc",
        #                                        (str(n) + ".jpg")))
        #          # cropped_img.show()
        #          n += 1
        fig.add_subplot(2, 5, 2)
        plt.imshow(img)
        #plt.show()
        plt.axis('off')
        for i in range(5):
            # img = Image.open(adresses[i][2].replace('\n',""))
            img = cv2.imread(
                os.path.join(r'E:\rov_data\update4_eilat', adresses[i][1].replace("/docker/bags/", "").replace('\n', "")))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((224, 224))
            fig.add_subplot(2, 5, i + 6)
            plt.imshow(img)
            plt.axis('off')
        #plt.show()
        file_name = str(j + 1)
        filename = os.path.join(args.save_path, 'best 5', file_name)
        plt.savefig(filename, bbox_inches='tight')
        # להכניס כאן את השוואת המרובעים
        #plt.show()

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

        [top1,similarity] = accuracy(query_features, reference_features, query_labels.astype(int))

    if args.evaluate:
        np.save(os.path.join(args.save_path, 'grd_global_descriptor.npy'), query_features)
        np.save('sat_global_descriptor.npy', reference_features)

    return top1, similarity


def save_checkpoint(state, is_best, filename='model_best_pretrained.pth', args=None):
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
            #print(similarity[i,:])
            #print(similarity[i, query_labels[i]])
            ranking = np.sum((similarity[i,:]>similarity[i,query_labels[i]])*1.)

            for j, k in enumerate(topk):
                if ranking < k:
                    results[j] += 1.


    results = results/ query_features.shape[0] * 100.
    print('Percentage-top1:{}, top5:{}, top10:{}, top1%:{}, time:{}'.format(results[0], results[1], results[2], results[-1], time.time() - ts))
    return results[:2], similarity

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
