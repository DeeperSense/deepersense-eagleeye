
import os
import pandas as pd
from Network.Train import validate
from Network.Train import validate_patches
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def best_5(val_FLC_loader, val_FLS_loader, model, args,test_id):

    adresses = []
    similarity = validate(val_FLC_loader, val_FLS_loader, model, args)[1]
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
        print(test_id[j][1])
        print(os.path.join(r'E:\rov_data\update4_eilat', test_id[j][1].replace("/docker/bags/", "").replace('\n', "")))
        img = cv2.imread(
            os.path.join(r'E:\rov_data\update4_eilat', test_id[j][1].replace("/docker/bags/", "").replace('\n', "")))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((224, 224))
        n = 0
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
        file_name = str(j + 1)
        filename = os.path.join(args.save_path, 'best 5', file_name)
        plt.savefig(filename, bbox_inches='tight')
        plt.show()


def Patch_match(dataloader, model, args):
    #similarity = validate_patches(quater_FLC_loader, quater_FLS_loader, model, args)[1]

    if not os.path.exists(os.path.join(args.save_path, 'patch_match')):
        os.mkdir(os.path.join(args.save_path, 'patch_match'))
    for k, (images_1, images_2) in enumerate(dataloader):
        similarity = validate_patches(images_1, images_2, model, args)[1]
        for j in range(len(similarity)):
            first_row = pd.DataFrame(similarity[j, :], columns=['col1'])  # .sort_values(by=['col1'], ascending=False)
            first_row = first_row.sort_values(by=['col1'], ascending=False)
            first_5 = first_row.head(4)
            indexes_5 = first_5.index.tolist()
            #adresses = [quater_id[i] for i in indexes_4]

            fig = plt.figure(figsize=(10, 3), dpi=420)
            plt.axis('off')
            img = Image.open( os.path.join(r"D:\fabian\PycharmProjects\Dual\results\patches\50425\flc",(str(k*4+ j)+".jpg")))
            #"D:\fabian\PycharmProjects\Dual\results\patches\2710\flc\0.jpg"
            img = img.resize((224, 224))
            fig.add_subplot(2, 5, 2)
            plt.imshow(img)
            # plt.show()
            plt.axis('off')
            for i in range(4):
                img = Image.open( os.path.join(r"D:\fabian\PycharmProjects\Dual\results\patches\50425\fls",(str(k*4+indexes_5[i])+".jpg")))
                img = img.resize((224, 224))
                fig.add_subplot(2, 5, i + 6)
                # plt.title(str(first_row['col1'].loc[indexes_4[i]]), fontsize=5)
                plt.imshow(img)
                # plt.show()
                plt.axis('off')

            file_name = str(k*4+ j + 1)
            filename = os.path.join(args.save_path, 'best quater', file_name)
            plt.savefig(filename, bbox_inches='tight')
            plt.show()
    print('end')