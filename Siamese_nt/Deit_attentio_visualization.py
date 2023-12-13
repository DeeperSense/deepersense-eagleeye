# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import pandas as pd
import torch
import torch.nn as nn
from functools import partial
from PIL import Image
from timm.models.vision_transformer import VisionTransformer, _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
import torchvision
import matplotlib.image
import matplotlib.pyplot as plt
import cv2
import csv


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, crop=False, save=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)
        self.save = save
        self.crop = crop
        self.crop_rate = 0.64 # keep rate: 0.53 352, 0.64 320, 0.79 288


    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        print("1")
        return x[:, 0], x[:, 1]

    def forward_features_save(self, x, indexes=None):
        def show_mask_on_image(img, mask):
            img = np.float32(img)/255
            img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            t = 0.6* np.max(mask) # Changed from 0.6
            #heatmap = cv2.applyColorMap(np.uint8(mask),cv2.COLORMAP_JET)
            #heatmap = np.float32(heatmap) / 255
            for i in range(0, 197, 28):
                r = i + 28
                indices = list(range(i, i + 28))
                strip = np.take(mask, indices, axis=1, mode='raise')
                #t = .8 * np.max(strip)
                ret, thresh = cv2.threshold(strip, t, 255, 0)

                # calculate moments of binary image
                M = cv2.moments(thresh)
                if M["m00"] > 0:
                    # calculate x,y coordinate of center
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # put text and highlight the center
                    cv2.circle(img, (cX + i, cY), 5, (255, 255, 255), -1)
                    # cv2.imshow("Image", img)
                    #cv2.waitKey()
            #cv2.putText(img, "FLS Object", (cX - 15, cY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255,255 ), 2)
            #cam = thresh + np.float32(img)
            #cam = cam / np.max(cam)
            # cam = heatmap + np.float32(img)
            # cam = cam / np.max(cam)
            return np.uint8(122*img) #np.uint8(255 * cam)


        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        x_shape = x.shape
        with open(r"E:\rov_data\update4_eilat\multi_501.csv") as f:
            my_csv_data = list(csv.reader(f))


        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if i == len(self.blocks)-1:  # len(self.blocks)-1:
                y = blk.norm1(x)
                B, N, C = y.shape
                qkv = blk.attn.qkv(y).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

                att = (q @ k.transpose(-2, -1)) * blk.attn.scale
                att = att.softmax(dim=-1)
                #att = (att @ v).transpose(1, 2)#.reshape(B, 1, C)


                last_map = (att[:, :, :2, 2:].detach().cpu().numpy()).sum(axis=1).sum(axis=1)
                last_map = last_map.reshape(
                    [last_map.shape[0],x_shape[2] // 16, x_shape[3] // 16])
            x = blk(x)

        for j, index in enumerate(indexes.cpu().numpy()):

            img = cv2.imread(os.path.join(r'E:\rov_data\update4_eilat', my_csv_data[index][4].replace('\n', ""),"imgs", my_csv_data[index][2] ))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img=img.resize((224,224))
            mask = cv2.resize(last_map[j], (224, 224))
            matplotlib.image.imsave(os.path.join(self.save, str(indexes[j].cpu().numpy()) + 'mask.png'),mask)
            fig = plt.figure(figsize=(10, 3), dpi=160)
            fig.add_subplot(1, 3, 1)
            plt.imshow(img)
            fig.add_subplot(1, 3, 2)
            plt.imshow(mask)
            fig.add_subplot(1, 3, 3)
            df=pd.DataFrame(mask)
            #df.to_csv(os.path.join(self.save, str(indexes[j].cpu().numpy()) + '.csv'))
            mask = show_mask_on_image(img, mask)
            plt.imshow(mask,cmap='gray' )#
            #plt.savefig(os.path.join(self.save, str(indexes[j].cpu().numpy()) + '.png'))
            #plt.show()

        x = self.norm(x)

        return x[:, 0], x[:, 1]

    def forward_features_crop(self, x, atten):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # add nonuniform-cropping

        B = x.shape[0]
        grid_size = (x.shape[-2] // self.patch_embed.patch_size[0], x.shape[-1] // self.patch_embed.patch_size[1])
        x = self.patch_embed(x)
        # sort based on attention
        atten_reshape = torch.nn.functional.interpolate(atten.detach(), grid_size, mode='bilinear')
        order = torch.argsort(atten_reshape[:,0,:,:].reshape([B,-1]),dim=1)
        # select patches
        select_list = []
        pos_list = []
        for k in range(B):
            select_list.append(x[k,order[[k],-int(self.crop_rate*order.shape[1]):]])
            pos_list.append(torch.cat([self.pos_embed[:,:2],self.pos_embed[:,2+order[k,-int(self.crop_rate*order.shape[1]):]]],dim=1))

        x = torch.cat(select_list,dim=0)
        pos_embed = torch.cat(pos_list,dim=0)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]


    def forward(self, x, atten=None, indexes=None):
        if self.save is not None:
            x, x_dist = self.forward_features_save(x, indexes)
        elif self.crop:
            if atten is None:
                atten = torch.zeros_like(x).cuda()
            x, x_dist = self.forward_features_crop(x, atten)
        else:
            x, x_dist = self.forward_features(x)

        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        # follow the evaluation of deit, simple average and no distillation during training, could remove the x_dist
        return (x + x_dist) / 2

@register_model
def deit_small_distilled_patch16_224(pretrained=True, img_size=(224,224), num_classes =1000, **kwargs):
    model = DistilledVisionTransformer(
        img_size=img_size, patch_size=16, embed_dim=384, num_classes=num_classes, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        # for key in checkpoint["model"]:
        #     print(key)

        # resize the positional embedding
        weight = checkpoint["model"]['pos_embed']
        ori_size = np.sqrt(weight.shape[1] - 1).astype(int)
        new_size = (img_size[0] // model.patch_embed.patch_size[0], img_size[1] // model.patch_embed.patch_size[1])
        matrix = weight[:, 2:, :].reshape([1, ori_size, ori_size, weight.shape[-1]]).permute((0, 3, 1, 2))
        resize = torchvision.transforms.Resize(new_size)
        new_matrix = resize(matrix).permute(0, 2, 3, 1).reshape([1, -1, weight.shape[-1]])
        checkpoint["model"]['pos_embed'] = torch.cat([weight[:, :2, :], new_matrix], dim=1)
        # change the prediction head if not 1000
        if num_classes != 1000:
            checkpoint["model"]['head.weight'] = checkpoint["model"]['head.weight'].repeat(5,1)[:num_classes, :]
            checkpoint["model"]['head.bias'] = checkpoint["model"]['head.bias'].repeat(5)[:num_classes]
            checkpoint["model"]['head_dist.weight'] = checkpoint["model"]['head.weight'].repeat(5,1)[:num_classes, :]
            checkpoint["model"]['head_dist.bias'] = checkpoint["model"]['head.bias'].repeat(5)[:num_classes]
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])
    return model

