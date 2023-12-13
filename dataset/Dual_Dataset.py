import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt

transform_list = [#transforms.Grayscale(),
    transforms.ColorJitter(brightness=0.2),
    transforms.RandomAdjustSharpness(0.5),
    transforms.RandomCrop(size=180)
]
def train_transform(size):#transforms.RandomChoice(transform_list),
    return transforms.Compose([

        # transforms.RandomResizedCrop(size=56),
        # transforms.Resize(size=224),
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])
def input_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])


class Dual_Dataset(torch.utils.data.Dataset):
    def __init__(self, mode = '', root = r'/workspaces/Dual/dataset/data', print_bool=False, polar = '',args=None): #CV-dataset
        super(Dual_Dataset, self).__init__()

        self.args = args
        self.root = root
        self.polar = polar
        self.mode = mode
        self.sat_size = [224, 224]
        self.sat_size_default = [224, 224]
        self.grd_size = [224, 224]

        if print_bool:
            print(self.sat_size, self.grd_size)

        self.sat_ori_size = [750, 750]
        self.grd_ori_size = [224, 1232]

        self.train_transform=train_transform(size=self.grd_size)
        self.transform_FLC = input_transform(size=self.grd_size)


        self.transform_FLS = input_transform(size=self.sat_size)
        self.train_list = os.path.join(self.root,'single_501.csv')
        self.test_list = os.path.join(self.root,'single_501.csv')
        # self.quater_list = os.path.join(self.root,'quater_set.csv')

        if print_bool:
            print('Dual: load %s' % self.train_list)
        self.__cur_id = 0  # for training
        self.id_list = []
        self.id_idx_list = []
        with open(self.train_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                self.id_list.append([data[1], os.path.join(data[4].replace('\n',""),"imgs",data[3] ), os.path.join(data[4].replace('\n',""),"imgs",data[2] )])  # update2
                self.id_idx_list.append(idx)
                idx += 1
        self.data_size = len(self.id_list)
        if print_bool:
            print('Dual: load', self.train_list, ' data_size =', self.data_size)
            print('Dual: load %s' % self.test_list)
        self.__cur_test_id = 0  # for training
        self.id_test_list = []
        self.id_test_idx_list = []
        with open(self.test_list, 'r') as file:
            idx = 0
            for line in file:
                data = line.split(',')
                #self.id_test_list.append([data[0], data[1],data[2]])# update1
                self.id_test_list.append([data[1], os.path.join(data[4].replace('\n', ""), "imgs", data[3]), os.path.join(data[4].replace('\n', ""), "imgs", data[2])])
                self.id_test_idx_list.append(idx)
                idx += 1
        self.test_data_size = len(self.id_test_list)
        self.__cur_test_id = 0  # for training
        # self.id_quater_list = []
        # self.id_quater_idx_list = []
        # with open(self.quater_list, 'r') as file:
        #      idx = 0
        #      for line in file:
        #          data = line.split(',')
        #          self.id_quater_list.append([data[0], data[1],data[2]])# pano_id
        #          self.id_quater_idx_list.append(idx)
        #          idx += 1
        # self.quater_list_data_size = len(self.id_quater_list)
        if print_bool:
            print('Dual: load', self.test_list, ' data_size =', self.test_data_size)

    def __getitem__(self, index, debug=False):
        print('mode', self.mode)
        if self.mode== 'train':
            idx = index % len(self.id_idx_list)
            img_FLC =cv2.imread(os.path.join(self.root, self.id_list[idx][2].replace("/docker/bags/","").replace('\n',"")))
            img_FLC=cv2.cvtColor(img_FLC, cv2.COLOR_BGR2RGB)
            img_FLC= Image.fromarray(img_FLC)
            img_FLS = cv2.imread(os.path.join(self.root, self.id_list[idx][1].replace("/docker/bags/", "")))
            img_FLS = cv2.cvtColor(img_FLS, cv2.COLOR_BGR2RGB)
            img_FLS = Image.fromarray(img_FLS)
            img_FLC = self.train_transform(img_FLC)
            img_FLS = self.train_transform(img_FLS)
            return img_FLC, img_FLS, torch.tensor(idx), torch.tensor(idx)

        elif 'scan_val' in self.mode:
            img_FLS =cv2.imread(os.path.join(self.root, self.id_test_list[index][1].replace("/docker/bags/","").replace('\n',"")))#self.root +
            img_FLS = cv2.cvtColor(img_FLS, cv2.COLOR_BGR2RGB)
            img_FLS = Image.fromarray(img_FLS)
            img_FLS = self.transform_FLS(img_FLS)
            # img_FLC = cv2.imread( os.path.join(self.root, self.id_test_list[index][2].replace("/docker/bags/", "").replace('\n', "")))
            # flc_path = os.path.join(self.root, self.id_test_list[index][1].replace("/docker/bags/", "").replace('\n', ""))
            flc_path = '/workspaces/Dual/dataset/data/20221211_092506/imgs/00019230_original.tiff'
            img_FLC = cv2.imread(flc_path)
        
            img_FLC = cv2.cvtColor(img_FLC, cv2.COLOR_BGR2RGB)
            img_FLC = Image.fromarray(img_FLC)
            img_FLC = self.transform_FLC(img_FLC)
            return img_FLC, img_FLS, torch.tensor(index), torch.tensor(index)#, 0, 0

        elif 'test_FLS' in self.mode:
            img_FLS = cv2.imread(os.path.join(self.root,self.id_test_list[index][1].replace("/docker/bags/", "").replace('\n', "")))  # self.root +
            img_FLS = cv2.cvtColor(img_FLS, cv2.COLOR_BGR2RGB)
            img_FLS = Image.fromarray(img_FLS)
            img_FLS = self.transform_FLS(img_FLS)

            return img_FLS, torch.tensor(index), #0

        elif 'test_FLC' in self.mode:
            img_FLC = cv2.imread(os.path.join(self.root, self.id_test_list[index][2].replace("/docker/bags/", "").replace('\n', "")))  # self.root +
            img_FLC = cv2.cvtColor(img_FLC, cv2.COLOR_BGR2RGB)
            img_FLC = Image.fromarray(img_FLC)
            img_FLC = self.transform_FLC(img_FLC)
            return img_FLC, torch.tensor(index), torch.tensor(index)

        elif 'quater_FLC' in self.mode:
            img_FLC = cv2.imread(os.path.join(r"D:\fabian\PycharmProjects\Dual\results\quarters\flc\flc", (str(index) + ".jpg")))
            img_FLC = cv2.cvtColor(img_FLC, cv2.COLOR_BGR2RGB)
            img_FLC = Image.fromarray(img_FLC)
            img_FLC = self.transform_FLC(img_FLC)
            return img_FLC, torch.tensor(index), torch.tensor(index)

        elif 'quater_FLS' in self.mode:
            img_FLS = cv2.imread(os.path.join(r"D:\fabian\PycharmProjects\Dual\results\quarters\fls\fls", (str(index) + ".jpg")))  # self.root +
            img_FLS = cv2.cvtColor(img_FLS, cv2.COLOR_BGR2RGB)
            img_FLS = Image.fromarray(img_FLS)
            img_FLS = self.transform_FLS(img_FLS)

            return img_FLS, torch.tensor(index), #0
        else:
            print('not implemented!!')
            raise Exception

    def __len__(self):
        if self.mode == 'train':
            return len(self.id_idx_list)
        elif 'scan_val' in self.mode:
            return len(self.id_test_list)
        elif 'test_FLS' in self.mode:
            return len(self.id_test_list)
        elif 'test_FLC' in self.mode:
            return len(self.id_test_list)
        elif 'quater_FLS' in self.mode:
            return len(self.id_quater_list)
        elif 'quater_FLC' in self.mode:
            return len(self.id_quater_list)
        else:
            print('not implemented!')
            raise Exception
