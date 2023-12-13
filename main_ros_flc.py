
# ROS
import rospy
from sensor_msgs.msg import Image
import cv2
import cv_bridge

import message_filters
import threading


import json
from std_msgs.msg import Int64MultiArray, MultiArrayDimension


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
parser.add_argument('--resume', default=r"/workspaces/Dual/modelweights/results_train5_test25_batch62/model_best.pth", type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_path', default=r'/workspaces/Dual/results', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--dim', default=1000, type=int,
                    help='feature dimension (default: 128)')



def main(mode):
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



    if mode == 'flc':
        model.FLC_net.save = os.path.join(args.save_path, 'attention', 'trainflc')
        return train_scan_loader, model, args        

    elif mode == 'fls':
        model.FLS_net.save = os.path.join(args.save_path, 'attention', 'trainfls')
        model.FLC_net.save = os.path.join(args.save_path, 'attention', 'trainfls')
        return val_scan_loader, model, args 



    # if args.attention:
    #     # FLC
    #     #model.FLS_net.save = os.path.join(args.save_path, 'attention', 'trainfls')
    #     model.FLC_net.save = os.path.join(args.save_path, 'attention', 'trainflc')
    #     scan(train_scan_loader, model, args, )
        
        
    #     # FLS
    #     model.FLS_net.save = os.path.join(args.save_path, 'attention', 'trainfls')
    #     model.FLC_net.save = os.path.join(args.save_path, 'attention', 'trainfls')
    #     scan(val_scan_loader, model, args)

if __name__ == '__main__':
    
    bridge = cv_bridge.CvBridge()
    rospy.init_node('dual_flc')

    pub_flc = rospy.Publisher('dual/FLC', Image, queue_size=10)
    pub_fls = rospy.Publisher('dual/FLS', Image, queue_size=10)

    pub_flc_points = rospy.Publisher('dual/FLC/points', Int64MultiArray, queue_size=10)
    pub_fls_points = rospy.Publisher('dual/FLS/points', Int64MultiArray, queue_size=10)
    pub_intersection_points = rospy.Publisher('dual/intersection/points', Int64MultiArray, queue_size=10)




    # FLC
    loader_flc, model_flc, args_flc =  main('flc')

    # FLS
    loader_fls, model_fls, args_fls =  main('fls')

    



    def create_points(data):
        # Initialize Int64MultiArray
        msg = Int64MultiArray()

        # Initialize dimensions (rows and columns)
        msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
        msg.layout.dim[0].label = "height"
        msg.layout.dim[1].label = "width"
        msg.layout.dim[0].size = len(data) #5 #  Assumes you have 5 points
        msg.layout.dim[1].size = 2 #  For x and y
        msg.layout.dim[0].stride = len(data)*2
        msg.layout.dim[1].stride = 2
        msg.layout.data_offset = 0

        # Now let's fill the data (for example)  
        # xy_points = [[10, 100], [20, 200], [30, 300], [40, 400], [50, 500]]
        # for i in range(5):  
        for i in range(len(data)):
            msg.data.append(data[i][0])
            msg.data.append(data[i][1])
          
        # Publish the points 
        return msg



    def callback_flc(msg):
        print("Received an Camera image!")
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # img[-300:] = 0
        height, width = img.shape[:2]
        img = img[:height-780, :width-15]

        # save image to file
        path_in = '/workspaces/Dual/dataset/data/20221211_092506/imgs/00019230.tiff'
        cv2.imwrite(path_in, img)

        # img = cv2.resize(img, (640, 480))
        # cv2.imshow("Image", img)
        # cv2.waitKey(1)
        scan(loader_flc, model_flc, args_flc, )

        # open image from file
        path_out = '/workspaces/Dual/results/attention/trainflc/0.png'
        img = cv2.imread(path_out)

        # Re-convert the processed image to Image msg and publish
        img_msg = bridge.cv2_to_imgmsg(img, encoding='bgr8')
        pub_flc.publish(img_msg)

        # Open hitpoints from json file
        path_json = '/workspaces/Dual/results/attention/trainflc/0.json'
        with open(path_json) as f:
            data = json.load(f)
        # print(data)

        # Publish hitpoints
        # Initialize Int64MultiArray
        msg = create_points(data)
          
        # Publish the points 
        pub_flc_points.publish(msg)        



    def callback_fls(msg):
        print("Received an Sonar image!")
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # reszie to 200x265
        img = cv2.resize(img, (200, 265))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img[-25:] = 0

        # flip image horizontally
        img = cv2.flip(img, 1)

        # save image to file
        path_in = '/workspaces/Dual/dataset/data/20221211_092506/imgs/00019231.tiff'
        cv2.imwrite(path_in, img)

        # img = cv2.resize(img, (640, 480))
        # cv2.imshow("Image", img)
        # cv2.waitKey(1)
        scan(loader_fls, model_fls, args_fls, )

        # open image from file
        path_out = '/workspaces/Dual/results/attention/trainfls/0.png'
        img = cv2.imread(path_out)

        # Re-convert the processed image to Image msg and publish
        img_msg = bridge.cv2_to_imgmsg(img, encoding='bgr8')
        pub_fls.publish(img_msg)

        # Open hitpoints from json file
        path_json = '/workspaces/Dual/results/attention/trainfls/0.json'
        with open(path_json) as f:
            data = json.load(f)

        # Publish hitpoints
        # Initialize Int64MultiArray
        msg = create_points(data)

        # Publish the points
        pub_fls_points.publish(msg)



    # rospy.Subscriber('/camera/image_raw', Image, callback_flc, queue_size=1)
    # rospy.Subscriber('/oculus/drawn_sonar', Image, callback_fls, queue_size=1)

    def process_FLC(image):
        # Process image with model 1
        callback_flc(image)


    
    def process_FLS(image):
        # Process image with model 2
        callback_fls(image)


    def callback(FLC_image, FLS_image):
        print('cbd')
        # process_FLC(FLC_image)
        # process_FLS(FLS_image)


        # Create threads
        t1 = threading.Thread(target=process_FLC, args=(FLC_image,))
        t2 = threading.Thread(target=process_FLS, args=(FLS_image,))

        # Start threads
        t1.start()
        t2.start()

        # Wait for threads to finish
        t1.join()
        t2.join()

        # Open hitpoints from camera json file
        path_json = '/workspaces/Dual/results/attention/trainflc/0.json'
        with open(path_json) as f:
            flc_data = json.load(f)
        # print(data)

        # Open hitpoints from sonar json file
        path_json = '/workspaces/Dual/results/attention/trainfls/0.json'
        with open(path_json) as f:
            fls_data = json.load(f)




        # # Assuming your lists are like this
        # list1 = ['x0', 'y0', 'x1', 'y1']
        # list2 = ['x1', 'y2', 'x3', 'y3']

        list1 = flc_data
        list2 = fls_data

        # Extract x elements from two lists
        x_list1 = [x for x in list1 if 'x' in x]
        x_list2 = [x for x in list2 if 'x' in x]

        # Convert the lists into Python sets
        set1 = set(x_list1)
        set2 = set(x_list2)

        # Do the intersection
        intersection = set1.intersection(set2)

        # Publish the intersection
        pub_intersection_points.publish(create_points(intersection))


        


    FLC_sub = message_filters.Subscriber('/camera/image_raw', Image)
    FLS_sub = message_filters.Subscriber('/oculus/drawn_sonar_rect', Image)

    queue_size = 1
    slop = 0.1
    ts = message_filters.ApproximateTimeSynchronizer([FLC_sub, FLS_sub], queue_size, slop, allow_headerless=True)
    ts.registerCallback(callback)





    rospy.spin()



