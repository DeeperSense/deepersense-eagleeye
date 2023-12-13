import torch
import cv2
import os
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.transforms.functional as fn
import csv
from PIL import Image
from os import walk
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
import wandb as wandb
import timm
from collections import OrderedDict
root = r'E:\rov_data\update4_eilat'
train_list = root + r'\single_501.csv'
with open(train_list, 'r') as file:
      for line in file:
          data = line.split(',')
          path= os.path.join(root,data[4].replace('\n',""),'imgs', str(data[3]))
          print((path))
          fig = plt.figure(figsize=(10, 3), dpi=420)
          img1 =cv2.imread(path)
          path = os.path.join(root, data[4].replace('\n', ""), 'imgs', str(data[2]))
          img2 = cv2.imread(path)
          img2=cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
          fig.add_subplot(1,2,2)
          plt.imshow(img1)
          plt.axis('off')
          fig.add_subplot(1, 2, 1)
          plt.imshow(img2)
          plt.axis('off')
          plt.show()
          print("end")




# image = Image.open(r"E:\rov_data\update4_eilat\20221211_092506\imgsO\00000001.tiff")
#
# # Get the size of the image
# width, height = image.size
#
# # Set the patch size
# patch_width = 28
# patch_height = 28
#
# # Calculate the number of patches in the horizontal and vertical directions
# num_patches_horizontal = width // patch_width
# num_patches_vertical = height // patch_height
#
# # Crop the image into patches
# patches = []
# for i in range(num_patches_vertical):
#     for j in range(num_patches_horizontal):
#         left = j * patch_width
#         upper = i * patch_height
#         right = left + patch_width
#         lower = upper + patch_height
#
#         patch = image.crop((left, upper, right, lower))
#         patches.append(patch)
#
# # Display the cropped patches
# for patch in patches:
#     patch.show()


#wandb.login()


 # %% lines from opher to upload checkpoint
#
# # path_to_checkoint = r"E:\dino_checkpoints\1\checkpoint.pth"
# # model = timm.create_model("vit_small_patch16_224_dino", pretrained=False)
# # model = timm.create_model("vit_small_patch16_224_dino", pretrained=False)
# #
# # checkpoint = torch.load(path_to_checkoint)
# #
#
# models = {
#     "supervised": timm.create_model("vit_small_patch16_224_dino", pretrained=True),
#     "selfsupervised": timm.create_model("vit_small_patch16_224_dino", pretrained=False)
# }
# new_state_dict = OrderedDict()
# state_dict = torch.load(r"E:\dino_checkpoints\1\checkpoint.pth")
#
# for k, v in state_dict["student"].items():
#     k = k.replace('module.backbone.', '')   # remove prefix backbone.
#     k = k.replace('attn.qkv.weight', 'attn.attn.in_proj_weight')
#     k = k.replace('attn.qkv.bias', 'attn.attn.in_proj_bias')
#     k = k.replace('attn.proj.weight', 'attn.attn.out_proj.weight')
#     k = k.replace('attn.proj.bias', 'attn.attn.out_proj.bias')
#     new_state_dict[k] = v
# state_dict["student"]=new_state_dict
# state_dict["teacher"]=new_state_dict
# print ("end")



# %%
#****Jason*****
# import json
# def init_dict_2(n):
#     ''' Initialize a dictionary with n key-value pairs. '''
#     return {i:str(i) for i in range(n)}
# d= init_dict_2(62)
# #print (d)
#
# json_object = json.dumps(d )
# with open("sample.json", "w") as outfile:
#     outfile.write(json_object)

# # # העברה מתיקיה לתיקיה
# root = r'E:\rov_data\update4_eilat'
# with open(r"E:\rov_data\update4_eilat\princess.csv", 'r') as file:
#       for line in file:
#           data = line.split(',')
#           path= os.path.join(root ,data[4],'imgs', str(data[3])) #.zfill(8)
#           img= cv2.imread(path)
#           # cv2.imshow("1", img)
#           # cv2.waitKey()
#           path1 = os.path.join(root, "sonar_for_princess", str(data[3]))
#           cv2.imwrite(path1, img)
#           print("end")


#**** מרכז מאסה****
#scipy.ndimage.center_of_mass(input, labels=None, index=None)
#https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
#*** center of gravity

#**** attention
# img = cv2.imread(r"D:\fabian\Result_Eilat\Eilat 14_2_23_sonar_straight\val\1190" r"mask.png")
#
# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# t = 0.5 * np.max(gray_image)
# ret, gray_image = cv2.threshold(gray_image, t, 255, 0)
# for i in range(0,197,28):
#     r=i+28
#     indices= list(range(i,i+28))
#     strip=np.take(gray_image, indices, axis=1, mode='raise')
#     # convert the grayscale image to binary image
#     #t=.8*np.max(strip)
#     ret, thresh = cv2.threshold(strip, t, 255, 0)
#
#     # calculate moments of binary image
#     M = cv2.moments(thresh)
#     if M["m00"]>0:
#         # calculate x,y coordinate of center
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])
#
#         # put text and highlight the center
#         cv2.circle(img, (cX+i, cY), 5, (255, 255, 255), -1)
#         cv2.imshow("Image",img)
# cv2.waitKey (0)

#***Sonar transfer***
# root = r'E:\rov_data\update4_eilat'
# train_list = root + r'\princess.csv'
#
# with open(train_list, 'r') as file:
#       idx = 0
#       for line in file:
#           data = line.split(',')
#           path= os.path.join(root,data[4], 'imgsO', str(data[3])) #.zfill(8)
#           #path=r"E:\rov_data\update4_eilat\presentation_images\8\00012106.tiff"
#           print((path + ".tiff"))
#           imgs= cv2.imread(path)
#           #imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
#           margin = 0.99  # Cut off the outer 10% of the image
#           # Do the polar rotation along 1024 angular steps with a radius of 256 pixels.
#           polar_img = cv2.resize(imgs, (544,300))
#           polar_img = cv2.warpPolar(polar_img, (300, 544), (272, 300), imgs.shape[1] * margin * 0.5, cv2.WARP_POLAR_LINEAR)
#           polar_img = polar_img[308:508, 35:300]
#           #polar_img = cv2.resize(polar_img, (224, 224))
#           # Rotate it sideways to be more visually pleasing
#           polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
#           cv2.imshow("2", polar_img)
#
#           #path1 = os.path.join(root, "sonar_for_princess", str(data[3]).zfill(8))
#           path1=r"C:\Users\fabian\Desktop\examples\1103atten_pointB.png"
#           cv2.imwrite(path1 , polar_img)
#           cv2.waitKey()
#
#           # path = os.path.join(root, data[4], 'imgs', str(data[2]).zfill(8))
#           # imgc = cv2.imread(path + ".tiff")
#           # #imgc = cv2.cvtColor(imgc, cv2.COLOR_BGR2RGB)
#           # path1 = os.path.join(root, data[4], 'imgC', str(data[2]).zfill(8))
#           # cv2.imwrite(path1 + ".tiff",imgc)
#           # #plt.imshow(imgc), plt.show()

# margin = 0.99 # Cut off the outer 10% of the image
# # Do the polar rotation along 1024 angular steps with a radius of 256 pixels.
# polar_img = cv2.warpPolar(image, (300,544), (272,300), image.shape[1]*margin*0.5, cv2.WARP_POLAR_LINEAR)
# polar_img= polar_img[308:508, 35:300]
# polar_img= cv2.resize(polar_img, (224,224))
# # Rotate it sideways to be more visually pleasing
# polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
# plt.imshow(polar_img),plt.show()



#***** prepare subgroups
# df=pd.read_csv(r"E:\rov_data\update4_eilat\test_set.csv")
# df1=pd.read_csv(r"E:\rov_data\update4_eilat\multi_50 .csv")
# sample_set=df.loc[df['A'].isin(df1['A'])]
# sample_set.to_csv(r"E:\rov_data\update4_eilat\multi_501.csv")


#
#*** downsample video
# df=pd.read_csv(r"E:\rov_data\update4_eilat\train_set.csv")
# df= df[df.index % 5 == 0]
# df.to_csv(r"E:\rov_data\update4_eilat\train_set5.csv")

# d=pd.DataFrame(columns=['accuracy'])
#
# for i in range (10):
#      j=pd.Series(i,index=['accuracy'])
#      d=d.append(j ,ignore_index=True)
#
# d.to_csv(r"D:\fabian\PycharmProjects\Dual\results\accuracy.csv")



"""for (dirpath, dirnames, filenames) in walk(r"D:\fabian\PycharmProjects\transgeo1\results\quarters\sonar"):
     print("Directory path: ", dirpath)
     print("Folder name: ", dirnames)
     print("File name: ", filenames)"""

# sift = cv2.xfeatures2d_SIFT.create()
# kp = sift.detect(img, None)
# imgT=cv2.drawKeypoints(img,kp,img)

#crop upper side, treshold,SIFT
# root = r'E:\rov_data\update4_eilat'
# train_list = root + r'\test_set.csv '
# with open(train_list, 'r') as file:
#      idx = 0
#      for line in file:
#          data = line.split(',')
#          print (line)
#          #fig = plt.figure(figsize=(10, 3), dpi=160)
#          img= cv2.imread( os.path.join(root,data[4], 'imgs',data[3]))
#          img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#          img = Image.fromarray(img)
#          area = (0,304,968,608)
#          #img = img.crop(area)
#          img = img.save(os.path.join(root, data[4], 'imgs1', data[3]))
#          #fig.add_subplot(1,2, 1)
#          #plt.imshow(img)
#          # plt.axis('off')
#          # imgs= cv2.imread( os.path.join(root,data[4], 'imgs',data[2]))
#          # imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
#          # imgs = Image.fromarray(imgs)
#          # fig.add_subplot(1, 2, 2)
#          # plt.imshow(imgs)
#          # plt.axis('off')

# img= cv2.imread(r"E:\rov_data\update4_eilat\20221211_092506\imgs\00021492.tiff")
#
# img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = Image.fromarray(img)
# plt.imshow(img)
# plt.show()

# #####prepare examples####
# fig = plt.figure(figsize=(10, 3), dpi=160)
# img = cv2.imread(r"C:\Users\fabian\Desktop\for Ana\00032399.tiff")
# dim = (224,224)
# resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# #cv2.rectangle(resized, (56, 56), (168, 168), (255, 0, 0))
# #cv2.imshow('image', resized)
# img = cv2.imread(r"C:\Users\fabian\Desktop\for Ana\00032398.tiff")
# dim = (224,224)
# resized1 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# #cv2.rectangle(resized1, (56, 10), (168, 220), (255, 0, 0))
# numpy_vertical = np.hstack((resized, resized1))
# cv2.imshow('image', numpy_vertical)
# cv2.imshow('image1', resized1)
# cv2.imwrite(r"C:\Users\fabian\Desktop\exmampels\clear on the right side!.tiff", numpy_vertical)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#***make quarters in the middle
# T=transforms.Compose([
#         transforms.Resize(size=(224,224)),
#         transforms.ToTensor()])
#
# img= cv2.imread(r"C:\Users\fabian\Desktop\exmampels\00014289.tiff")
# img=cv2.resize(img,(224,224))
# img=img[56:168, 56:168]
# img=cv2.resize(img,(224,224))
# cv2.imwrite(r"D:\fabian\PycharmProjects\Dual\results\quarters\flc\flc\0.jpg",img)
# # plt.imshow(img.permute(1, 2, 0))
# # plt.show()
# cv2.imshow("1",img )
# cv2.waitKey()
print("end")


# root = r'E:\rov_data\update4_eilat'
# train_list = root + r'\multi_50 .csv '
# with open(train_list, 'r') as file:
#      idx = 0
#      for line in file:
#          data = line.split(',')
#          print (line)
#          img= cv2.imread( os.path.join(root,data[4], 'imgs',data[2]))
#          cv2.imshow("1",img)
#          cv2.waitKey()