import argparse
import os
import pandas as pd
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from datetime import datetime
#import datetime
# import wandb

from dataset.Dual_Dataset import Dual_Dataset
from dataset.Patch_Dataset import Patch_dataset,ImagePairDataset
from Siamese_nt.Dual_Model import Dual_Model
from Network.Cost_functions import SoftTripletBiLoss
from Network.Train import train
from Network.Train import validate
from Network.Train import accuracy
from Network.Train import scan
from Network.Train import adjust_learning_rate
from Network.Train import save_checkpoint
from  Network.Inference import best_5
from  Network.Inference import Patch_match
from torch.utils.data import  DataLoader
import torchvision.transforms as transforms
# import wandb as wandb
# %%
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-e', '--evaluate',default=False, dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-BF', '--Best_5',default=False, dest='Best_5', action='store_true',
                    help='Store Best_5 matching')
parser.add_argument('-MP', '--Make_patch',default=False, dest='Make_patch', action='store_true',
                    help='Store Make patch')
parser.add_argument('-PM', '--Patch_match',default=False, dest='Patch_match', action='store_true',
                    help='Store patch matching')
parser.add_argument('-tr', '--train',default=False, dest='train', action='store_true',
                    help='Store train or not')
parser.add_argument('-at', '--attention',default=True, dest='attention', action='store_true',
                    help='Store compute and save attention or not')
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--start-epoch', default=0, type=int, help='define start epoch number')
parser.add_argument('-b', '--batch-size', default=32, type=int)

parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, dest='lr')
parser.add_argument('--cos', action='store_true',default=False,
                    help='use cosine lr schedule')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=r"/workspaces/Dual/modelweights/results_train5_test25_batch 62/model_best.pth", type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_path', default=r'/workspaces/Dual/results', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--dim', default=1000, type=int,
                    help='feature dimension (default: 128)')



def main():
    #wandb.init(project='Dual')

    global best_accuracy
    best_accuracy = 0
    args = parser.parse_args()

    type_run = "train"  # "eval"
    # wandb.init(project='Dual', name=f"{str(datetime.now())}_{type_run}_5_test25_alfa30", config=args)


    gpu=args.gpu
    torch.cuda.set_device(gpu)
    model = Dual_Model(args=args)
    model = model.cuda(gpu)
    Cost_function = SoftTripletBiLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_accuracy = checkpoint['best_accuracy']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
        os.mkdir(os.path.join(args.save_path, 'attention'))
        os.mkdir(os.path.join(args.save_path, 'attention', 'train'))
        os.mkdir(os.path.join(args.save_path, 'attention', 'val'))
    dataset = Dual_Dataset
    train_dataset = dataset(mode='train', print_bool=True, args=args)
    val_scan_dataset = dataset(mode='scan_val', args=args)
    val_FLC_dataset = dataset(mode='test_FLC', args=args)
    val_FLS_dataset = dataset(mode='test_FLS', args=args)
    quater_FLC_datatset= dataset(mode='quater_FLC',  args=args)
    quater_FLS_dataset = dataset(mode='quater_FLS',  args=args)

    quater_FLC_loader = torch.utils.data.DataLoader(
        quater_FLC_datatset, batch_size=4, shuffle=False,
         pin_memory=True)
    quater_FLS_loader = torch.utils.data.DataLoader(
        quater_FLS_dataset, batch_size=4, shuffle=False,
        pin_memory=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
         pin_memory=True,  drop_last=False)

    train_scan_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
         pin_memory=True,
        drop_last=False)

    val_scan_loader = torch.utils.data.DataLoader(
        val_scan_dataset, batch_size=8, shuffle=False,
         pin_memory=True,
        drop_last=False)

    val_FLC_loader = torch.utils.data.DataLoader(
        val_FLC_dataset, batch_size=8, shuffle=False,
         pin_memory=True)
    val_FLS_loader = torch.utils.data.DataLoader(
        val_FLS_dataset, batch_size=8, shuffle=False,
        pin_memory=True)


    if args.evaluate:
        validate(val_FLC_loader, val_FLS_loader, model, args)

    if args.Best_5:
        args.evaluate = True
        test_id = val_FLC_dataset.id_test_list
        best_5(val_FLC_loader, val_FLS_loader, model, args,test_id)

    if args.Make_patch:
        Patches = Patch_dataset()
        Patches.make_patches()


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), ])
    root_dir_1= r"/workspaces/Dual/results/patches/50425/flc"
    root_dir_2=r"/workspaces/Dual/results/patches/50425/fls"
    if args.Patch_match:
        dataset=ImagePairDataset( root_dir_1, root_dir_2, transform= transform)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        args.evaluate = True
        Patch_match(dataloader, model, args)

    if args.train:
        PD = pd.DataFrame(columns=['accuracy', 'loss'])
        for epoch in range(args.start_epoch, args.epochs):
            print('start epoch:{}, date:{}'.format(epoch, datetime.now()))
            adjust_learning_rate(optimizer, epoch, args)
            train(train_loader, model, Cost_function, optimizer, epoch, args)
            accuracy = validate(val_FLC_loader, val_FLS_loader, model, args)
            # wandb.log({"Accuracy": accuracy[0]})
            r = pd.Series(accuracy[0] , index=['accuracy'])
            PD = PD.append(r, ignore_index=True)
            PD.to_csv(r"/workspaces/Dual/results/accuracy.csv")

            # remember best acc@1 and save checkpoint
            is_best = accuracy[0] > best_accuracy
            best_accuracy = max(accuracy[0], best_accuracy)
            # torch.distributed.barrier()

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_accuracy': best_accuracy,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename='checkpoint.pth', args=args)
    if args.attention:
        #model.FLS_net.save = os.path.join(args.save_path, 'attention', 'trainfls')
        model.FLC_net.save = os.path.join(args.save_path, 'attention', 'trainflc')
        scan(train_scan_loader, model, args, )
        model.FLS_net.save = os.path.join(args.save_path, 'attention', 'trainfls')
        model.FLC_net.save = os.path.join(args.save_path, 'attention', 'trainfls')
        scan(val_scan_loader, model, args)

if __name__ == '__main__':
    main()
