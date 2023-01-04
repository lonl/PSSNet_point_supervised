import torch.nn as nn
import torchvision
import torch
import os
from torchvision import models
from skimage import morphology as morph
from skimage import data, util
from skimage.io import imread,imsave
from skimage import data,segmentation,measure,morphology,color
import matplotlib.patches as mpatches
import numpy as np
import utils as ut
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

class BaseModel(nn.Module):
    def __init__(self, n_classes):
        super(BaseModel,self).__init__()
        self.trained_images = set()
        self.n_classes = n_classes

    @torch.no_grad()
    def predict(self, batch, method="probs"):
        self.eval()                 #predict
        # print('self.eval(): ',self.eval())
        if method == "counts":
            images = batch["images"].cuda()
            # pred_mask = self(images).data.max(1)[1].squeeze().cpu().numpy()
            origin_mask = F.softmax(self(images),1)                                 #(1,2,500,500)
            # print('origin_mask: ',origin_mask)
            mask_numpy = ut.t2n(origin_mask[0])                                     #(2,500,500)
            h_mask,w_mask = mask_numpy[1].shape
            for i in range(h_mask):
                for j in range(w_mask):
                    if mask_numpy[1][i][j] > 0.15:
                        mask_numpy[1][i][j] = 1.0
            pred_mask = np.argmax(mask_numpy,axis=0)

            counts = np.zeros(self.n_classes-1)

            for category_id in np.unique(pred_mask):
                if category_id == 0:
                    continue
                blobs_category = morph.label(pred_mask==category_id)
                # print('blobs_category:{}\n| np.unique(blobs_category):{}'.format(blobs_category,np.unique(blobs_category)))
                # print('(np.unique(blobs_category) != 0): ',(np.unique(blobs_category) != 0))
                n_blobs = (np.unique(blobs_category) != 0).sum()
                counts[category_id-1] = n_blobs
                print('counts[None]: ',counts[None])

            return counts[None]

        elif method == "blobs":

            images = batch["images"].cuda()
            # print('batch: ',batch)

            # print('images.shape: ',images.shape)                                    #torch.Size([1, 3, 500, 500])
            # print('self(images): ',self(images))                                    #torch.Size([1, 2, 500, 500])
            # print('self(images): ',self(images).data)
            # print('self(images).data.max(1): ',self(images).data.max(1))
            # pred_mask = self(images).data.max(1)[1].squeeze().cpu().numpy()           #(500,500)
            #
            # plt.imsave(os.path.join('figures/pred_mask/' + name + '_mask.png'), pred_mask, cmap='gray')

#fixme:softmax pred_mask
            origin_mask = F.softmax(self(images),1)                                 #(1,2,500,500)
            # print('origin_mask: ',origin_mask)
            mask_numpy = ut.t2n(origin_mask[0])                                     #(2,500,500)
            # print('mask_numpy_origin: ',mask_numpy)
            # np.savetxt('mask_froeground_before.txt',mask_numpy[1])
            #foreground
            h_mask,w_mask = mask_numpy[1].shape
            for i in range(h_mask):
                for j in range(w_mask):
                    if mask_numpy[1][i][j] > 0.15:
                        mask_numpy[1][i][j] = 1.0
            # np.savetxt('mask_froeground_after.txt', mask_numpy[1])
            # print('mask_numpy_after: ', mask_numpy)
            pred_mask = np.argmax(mask_numpy,axis=0)
            # print('pred_mask.shape: ',pred_mask.shape)
            # plt.imsave(os.path.join('figures/softmax_mask/'+name+'_mask.png'),pred_mask,cmap='gray')


            # print('torch.max(self(images).data): ',torch.max(self(images).data,dim=1))

            h,w = pred_mask.shape
            blobs = np.zeros((self.n_classes-1, h, w), int)
            # print('np.unique(pred_mask):',np.unique(pred_mask))

            for category_id in np.unique(pred_mask):    #[0 1]
                if category_id == 0:
                    continue
                # print('pred_mask == category_id :',pred_mask == category_id)
                blobs[category_id-1] = morph.label(pred_mask==category_id,connectivity=1)
                # print('model_blobs: ',blobs)
                # print('np.unique(blobs) != 0: ',(np.unique(blobs) != 0))
            # np.savetxt('blob.txt',blobs.squeeze())

            return blobs[None]

##------------------------------------------unet------------------


#----------------------------------------------------------------------#

def unetUp(inputs1,inputs2):
    offset_H = inputs1.size()[2] - inputs2.size()[2]
    offset_W = inputs1.size()[3] - inputs2.size()[3]
    padding = [0, offset_W, 0, offset_H]
    outputs2 = F.pad(inputs2,padding)
    return torch.cat((inputs1,outputs2), 1)


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class ResUnet2(BaseModel):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet50(https://arxiv.org/abs/1512.03385) encoder

        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/

        """

    def __init__(self, n_classes, num_filters=128, pretrained=True, is_deconv=True):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet50
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super(ResUnet2,self).__init__(n_classes)
        self.num_classes = n_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet50(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.dec4 = DecoderBlockV2(2048, 2048, 1024, is_deconv)
        self.dec3 = DecoderBlockV2(2048, 1024, 512, is_deconv)
        self.dec2 = DecoderBlockV2(1024, 512, 256, is_deconv)
        self.dec1 = DecoderBlockV2(512, 256, 64, is_deconv)
        self.dec0 = DecoderBlockV2(128, 64, 32, is_deconv)
        self.final = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):

        input_spatial_dim = x.size()[2:]
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        dec4 = self.dec4(conv5)

        dec3 = self.dec3(unetUp(dec4, conv4))
        dec2 = self.dec2(unetUp(dec3, conv3))
        dec1 = self.dec1(unetUp(dec2, conv2))
        dec0 = self.dec0(unetUp(dec1,conv1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)

        x_out = nn.functional.interpolate(x_out,
                                        size=input_spatial_dim,
                                        mode="bilinear",
                                        align_corners=True)

        return x_out
    



        