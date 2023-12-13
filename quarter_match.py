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
from dataset.Dual_Dataset import Dual_Dataset
from Siamese_nt.Dual_Model import Dual_Model
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms
from torchvision import datasets
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
parser.add_argument('-b', '--batch-size', default=4, type=int,
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
parser.add_argument('--resume', default=r"D:\fabian\PycharmProjects\Dual\results_9_4_sonar_straight_03lr\model_best.pth", type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_path', default=r'D:\fabian\PycharmProjects\Dual\results', type=str, metavar='PATH',
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
                    help='feature dimension (default: 124)')

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


best_accuracy = 0
size=(224,224)
def input_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.445, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])

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
    global best_accuracy

    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node
    print("Use GPU: {} for training".format(args.gpu))
    print("=> creating model '{}'")

    model = Dual_Model(args=args)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)



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
            best_accuracy= checkpoint['best_accuracy']
            model.load_state_dict(checkpoint['state_dict'],)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    dataset = Dual_Dataset


    quater_FLC_datatset= dataset(mode='quater_FLC',  args=args)
    quater_FLS_dataset = dataset(mode='quater_FLS',  args=args)

    train_sampler = None


    quater_FLC_loader = torch.utils.data.DataLoader(
        quater_FLC_datatset, batch_size=4, shuffle=False,
        num_workers=args.workers, pin_memory=True)  # 512, 64
    quater_FLS_loader = torch.utils.data.DataLoader(
        quater_FLS_dataset, batch_size=4, shuffle=False,
        num_workers=args.workers, pin_memory=True)  # 40, 124
    args.evaluate= True
    quater_id= quater_FLC_datatset.id_quater_list
    if args.evaluate:
        adresses=[]
        similarity=validate(quater_FLC_loader, quater_FLS_loader, model, args)[1]
        if not os.path.exists(os.path.join(args.save_path, 'best 5')):
            os.mkdir(os.path.join(args.save_path, 'best 5'))
        for j in range (len(quater_id)):
            first_row=pd.DataFrame(similarity[j,:], columns=['col1'])#.sort_values(by=['col1'], ascending=False)
            first_row= first_row.sort_values(by=['col1'], ascending=False)
            first_4=  first_row.head(4)
            indexes_4= first_4.index.tolist()
            adresses= [quater_id[i] for i in indexes_4]

            fig = plt.figure(figsize=(10, 3), dpi=420)
            plt.axis('off')
            img = cv2.imread(os.path.join(r'E:\rov_data\update4_eilat',quater_id[j][1].replace("/docker/bags/", "").replace('\n', "")))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((224, 224))
            fig.add_subplot(2,5, 2)
            plt.imshow(img)
            # plt.show()
            plt.axis('off')
            for i in range(4):
                #img = Image.open(adresses[i][2].replace('\n',""))
                # print( first_row['col1'].loc[indexes_4[i]])
                # print(os.path.join(r'E:\rov_data\update4_eilat', adresses[i][1].replace("/docker/bags/", "").replace('\n', "")))
                img = cv2.imread(os.path.join(r'E:\rov_data\update4_eilat', adresses[i][2].replace("/docker/bags/", "").replace('\n', "")))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = img.resize((224, 224))
                fig.add_subplot(2, 5, i + 6)
                #plt.title(str(first_row['col1'].loc[indexes_4[i]]), fontsize=5)
                plt.imshow(img)
                #plt.show()
                plt.axis('off')

            file_name=str(j+1)
            filename= os.path.join(args.save_path, 'best quater',file_name)
            plt.savefig(filename, bbox_inches='tight')
            plt.show()
    print('end')




# FLC features and FLS features should be computed separately without correspondence label
def validate(quater_FLC_loader, quater_FLS_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    progress_q = ProgressMeter(
        len(quater_FLC_loader),
        [batch_time],
        prefix='quater_FLC: ')
    progress_k = ProgressMeter(
        len(quater_FLS_loader),
        [batch_time],
        prefix='quater_FLS: ')

    # switch to evaluate mode
    model_FLC = model.FLC_net
    model_FLS = model.FLS_net
    model_FLC.eval()
    model_FLS.eval()
    print('model validate on cuda', args.gpu)

    FLC_features = np.zeros([len(quater_FLC_loader.dataset), args.dim])
    FLC_labels = np.zeros([len(quater_FLC_loader.dataset)])
    FLS_features = np.zeros([len(quater_FLS_loader.dataset), args.dim])

    with torch.no_grad():
        end = time.time()


        # end = time.time()

        # FLC features
        for i, (images, indexes, labels) in enumerate(quater_FLC_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                indexes = indexes.cuda(args.gpu, non_blocking=True)
                labels = labels.cuda(args.gpu, non_blocking=True)

            # compute output
            FLC_embed = model_FLC(images)

            FLC_features[indexes.cpu().numpy(), :] = FLC_embed.cpu().numpy()
            FLC_labels[indexes.cpu().numpy()] = labels.cpu().numpy()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress_q.display(i)

         # FLS features
        for i, (images, indexes) in enumerate(quater_FLS_loader):#, atten
             if args.gpu is not None:
                 images = images.cuda(args.gpu, non_blocking=True)
                 indexes = indexes.cuda(args.gpu, non_blocking=True)

            # compute output
            #  if args.crop:
            #      FLS_embed = model_FLS(x=images, atten=atten)
             #else:
             FLS_embed = model_FLS(x=images, indexes=indexes)  # delta

             FLS_features[indexes.cpu().numpy().astype(int), :] = FLS_embed.detach().cpu().numpy()

             # measure elapsed time
             batch_time.update(time.time() - end)
             end = time.time()

             if i % args.print_freq == 0:
                 progress_k.display(i)

        [top1,similarity] = accuracy(FLC_features, FLS_features, FLC_labels.astype(int))

    if args.evaluate:
        np.save(os.path.join(args.save_path, 'grd_global_descriptor.npy'), FLC_features)
        np.save('sat_global_descriptor.npy', FLS_features)

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


def accuracy(FLC_features, FLS_features, FLC_labels, topk=[1,5,10]):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    ts = time.time()
    N = FLC_features.shape[0]
    M = FLS_features.shape[0]
    topk.append(M//100)
    results = np.zeros([len(topk)])
    # for CVUSA, CVACT
    if N < 40000:
        FLC_features_norm = np.sqrt(np.sum(FLC_features**2, axis=1, keepdims=True))
        FLS_features_norm = np.sqrt(np.sum(FLS_features ** 2, axis=1, keepdims=True))
        similarity = np.matmul(FLC_features/FLC_features_norm, (FLS_features/FLS_features_norm).transpose())

        for i in range(N):
            #print(similarity[i,:])
            #print(similarity[i, FLC_labels[i]])
            ranking = np.sum((similarity[i,:]>similarity[i,FLC_labels[i]])*1.)

            for j, k in enumerate(topk):
                if ranking < k:
                    results[j] += 1.


    results = results/ FLC_features.shape[0] * 100.
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
