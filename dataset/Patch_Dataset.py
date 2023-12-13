import io
import torchvision.transforms as T
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
import os
import argparse
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

#cropping im1 = im.crop((left, top, right, bottom))
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--save_path', default=r'/workspaces/Dual/results', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
args = parser.parse_args()

class Patch_dataset():
    def __init__(self,root = r'E:\rov_data\update4_eilat'):
        self.root= root
        self.__cur_id = 0  # for training
        self.id_list = []
        self.id_idx_list = []
        self.test_list = self.root + r'\single_501.csv'
        with open(self.test_list, 'r') as file:
            idx = 0
            for line in file:
                self.data = line.split(',')
                self.id_list.append([self.data[1], os.path.join(self.data[4].replace('\n', ""), "imgs", self.data[3]),
                                     os.path.join(self.data[4].replace('\n', ""), "imgs", self.data[2])])  # update2
                self.id_idx_list.append(idx)
                idx += 1

    if not os.path.exists(os.path.join(args.save_path, 'patches')):
        os.makedirs(os.path.join(args.save_path, 'patches'))
    def make_patches(self):
        for j in range(len (self.id_idx_list)):
            if not os.path.exists(os.path.join(args.save_path, 'patches',self.id_list[j][0])):
                os.mkdir(os.path.join(args.save_path, 'patches',self.id_list[j][0]))
            if not os.path.exists(os.path.join(args.save_path, 'patches', self.id_list[j][0],'flc')):
                os.mkdir(os.path.join(args.save_path, 'patches', self.id_list[j][0],'flc'))
            if not os.path.exists(os.path.join(args.save_path, 'patches', self.id_list[j][0], 'fls')):
                os.mkdir(os.path.join(args.save_path, 'patches', self.id_list[j][0], 'fls'))

            img = Image.open(os.path.join(r'E:\rov_data\update4_eilat', self.id_list[j][2].replace("/docker/bags/", "").replace('\n', "")))
            img = img.resize((224, 224))
            # Get the size of the image
            width, height = img.size

            # Set the patch size
            patch_width = 56
            patch_height = 56

            # Calculate the number of patches in the horizontal and vertical directions
            num_patches_horizontal = width // patch_width
            num_patches_vertical = height // patch_height

            # Crop the image into patches
            n=0
            for i in range(num_patches_horizontal):
                for k in range(num_patches_vertical):
                    left = i * patch_width
                    upper = k * patch_height
                    right = left + patch_width
                    lower = upper + patch_height

                    cropped_img = img.crop((left, upper, right, lower))
                    cropped_img.save(os.path.join(args.save_path, 'patches', self.id_list[j][0], 'flc', (str(n) + ".jpg")))
                    print(os.path.join(args.save_path, 'patches', self.id_list[j][0], 'flc', (str(n) + ".jpg")))
                    n+=1

            img = Image.open(os.path.join(r'E:\rov_data\update4_eilat',self.id_list[j][1].replace("/docker/bags/", "").replace('\n', "")))
            img = img.resize((224, 224))
            # Get the size of the image
            width, height = img.size

            # Set the patch size
            patch_width = 56
            patch_height = 56

            # Calculate the number of patches in the horizontal and vertical directions
            num_patches_horizontal = width // patch_width
            num_patches_vertical = height // patch_height

            # Crop the image into patches
            n = 0
            for i in range(num_patches_horizontal):
                for k in range(num_patches_vertical):
                    left = i * patch_width
                    upper = k * patch_height
                    right = left + patch_width
                    lower = upper + patch_height

                    cropped_img = img.crop((left, upper, right, lower))
                    cropped_img.save(
                        os.path.join(args.save_path, 'patches', self.id_list[j][0], 'fls', (str(n) + ".jpg")))
                    print(os.path.join(args.save_path, 'patches', self.id_list[j][0], 'fls', (str(n) + ".jpg")))
                    n += 1


class ImagePairDataset(Dataset):
    def __init__(self, root_dir_1, root_dir_2, transform=None):
        self.root_dir_1 = root_dir_1
        self.root_dir_2 = root_dir_2
        self.transform = transform
        self.image_list_1 = os.listdir(root_dir_1)
        self.image_list_2 = os.listdir(root_dir_2)

    def __len__(self):
        return min(len(self.image_list_1), len(self.image_list_2))

    def __getitem__(self, idx):
        img_name_1 = os.path.join(self.root_dir_1, self.image_list_1[idx])
        img_name_2 = os.path.join(self.root_dir_2, self.image_list_2[idx])

        image_1 = Image.open(img_name_1)
        image_2 = Image.open(img_name_2)

        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        return image_1, image_2

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),])
    root_dir_1= r"D:\fabian\PycharmProjects\Dual\results\patches\2710\flc"
    root_dir_2=r"D:\fabian\PycharmProjects\Dual\results\patches\2710\fls"
    dataset=ImagePairDataset( root_dir_1, root_dir_2, transform= transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    for images_1, images_2 in dataloader:
        # Convert tensors to numpy arrays
        images_1 = images_1.numpy()
        images_2 = images_2.numpy()

        # Iterate through each pair of images
        for i in range(len(images_1)):
            # Display the first image
            plt.subplot(1, 2, 1)
            plt.imshow(np.transpose(images_1[i], (1, 2, 0)))
            plt.title('Image 1')
            plt.axis('off')

            # Display the second image
            plt.subplot(1, 2, 2)
            plt.imshow(np.transpose(images_2[i], (1, 2, 0)))
            plt.title('Image 2')
            plt.axis('off')

            # Show the images
        plt.show()
        with torch.no_grad():
            for i, (images_1, images_2) in enumerate(dataloader):  # atten
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                    indexes = indexes.cuda(args.gpu, non_blocking=True)
    print('end')

if __name__ == '__main__':
        main()




# mask using using https://sparrow.dev/torchvision-transforms/
"""preprocess=T.transforms.Compose([
        T.transforms.Resize(size=(224,224)),
        T.transforms.ToTensor(),
        #T.Normalize(
        #mean=[0.485, 0.456, 0.406],
        #std=[0.229, 0.224, 0.225])
])


class TopLeftCornerErase:
    def __init__(self, n_pixels: 16):
        self.n_pixels = 32

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        all_pixels = img.reshape(3, -1).transpose(1, 0)
        idx = torch.randint(len(all_pixels), (1,))[0]
        print (idx)
        random_pixel = all_pixels[idx][:, None, None]
        return F.erase(img, 0, 0, self.n_pixels, self.n_pixels, random_pixel)
reverse_preprocess = T.Compose([
    T.ToPILImage(),
    np.array,
])

x = preprocess(img)
erase = T.Compose([
    TopLeftCornerErase(100),
    reverse_preprocess,
])
x=erase(x)
plt.imshow(x)
plt.show()
print(x.shape)"""
