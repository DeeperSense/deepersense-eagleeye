import torch.nn as nn
from .Deit import deit_small_distilled_patch16_224

class Dual_Model(nn.Module):

    def __init__(self,  args, base_encoder=None):
        super(Dual_Model, self).__init__()
        self.dim = args.dim
        self.size_FLS = [224,224]#[256,478]
        self.size_FLC = [224, 224]
        base_model = deit_small_distilled_patch16_224
        self.FLC_net = base_model(img_size=self.size_FLC, num_classes=args.dim)
        self.FLS_net = base_model(img_size=self.size_FLS, num_classes=args.dim)

    def forward(self, im_q, im_k,indexes=None ):#
        return self.FLC_net(x=im_q ,indexes=indexes), self.FLS_net(x=im_k, indexes=indexes)# indexes=indexes
