import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from graph import GraphReasoning


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # conv1, 2 layers
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # conv2, 2 layers
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # conv3, 4 layers
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_4 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # conv4, 4 layers
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # conv5, 4 layers
        dila = [2, 4, 8, 16]
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=dila[0], dilation=dila[0])
        self.bn5_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=dila[1], dilation=dila[1])
        self.bn5_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=dila[2], dilation=dila[2])
        self.bn5_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.conv5_4 = nn.Conv2d(512, 512, 3, padding=dila[3], dilation=dila[3])
        self.bn5_4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_4 = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x  # [3, 256, 256]
        h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        h = self.relu1_2(self.bn1_2(self.conv1_2(h)))
        h_nopool1 = h
        h = self.pool1(h)
        h1 = h_nopool1              # [64, 256, 256]
        h = self.relu2_1(self.bn2_1(self.conv2_1(h)))
        h = self.relu2_2(self.bn2_2(self.conv2_2(h)))
        h_nopool2 = h
        h = self.pool2(h)
        h2 = h_nopool2              # [128, 128, 128]
        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))
        h = self.relu3_3(self.bn3_3(self.conv3_3(h)))
        h = self.relu3_4(self.bn3_4(self.conv3_4(h)))
        h_nopool3 = h
        h = self.pool3(h)
        h3 = h_nopool3              # [256, 64, 64]
        h = self.relu4_1(self.bn4_1(self.conv4_1(h)))
        h = self.relu4_2(self.bn4_2(self.conv4_2(h)))
        h = self.relu4_3(self.bn4_3(self.conv4_3(h)))
        h = self.relu4_4(self.bn4_4(self.conv4_4(h)))
        h_nopool4 = h
        #h = self.pool4(h)
        #h4 = h_nopool4             # [512, 32, 32]
        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
        h = self.relu5_3(self.bn5_3(self.conv5_3(h)))
        h = self.relu5_4(self.bn5_4(self.conv5_4(h)))
        h5 = h                      # [512, 32, 32]
        return h5, h3, h2  #h4 h1 

    def copy_params_from_vgg19_bn(self, vgg19_bn):
        features = [
            self.conv1_1, self.bn1_1, self.relu1_1,
            self.conv1_2, self.bn1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.bn2_1, self.relu2_1,
            self.conv2_2, self.bn2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.bn3_1, self.relu3_1,
            self.conv3_2, self.bn3_2, self.relu3_2,
            self.conv3_3, self.bn3_3, self.relu3_3,
            self.conv3_4, self.bn3_4, self.relu3_4,
            self.pool3,
            self.conv4_1, self.bn4_1, self.relu4_1,
            self.conv4_2, self.bn4_2, self.relu4_2,
            self.conv4_3, self.bn4_3, self.relu4_3,
            self.conv4_4, self.bn4_4, self.relu4_4,
            self.pool4,
            self.conv5_1, self.bn5_1, self.relu5_1,
            self.conv5_2, self.bn5_2, self.relu5_2,
            self.conv5_3, self.bn5_3, self.relu5_3,
            self.conv5_4, self.bn5_4, self.relu5_4,
        ]
        for l1, l2 in zip(vgg19_bn.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            if isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data


class CGR(nn.Module):
    def __init__(self, n_class=2, n_iter=2, chnn_side=(512, 256, 128), chnn_targ=(512, 128, 32, 4), rd_sc=32, dila=(4, 8, 16)):
        super().__init__()
        self.n_graph = len(chnn_side)
        n_node = len(dila)
        graph = [GraphReasoning(ii, rd_sc, dila, n_iter) for ii in chnn_side]
        self.graph = nn.ModuleList(graph)
        C_cat = [nn.Sequential(
            nn.Conv2d(ii//rd_sc*n_node, ii//rd_sc, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ii//rd_sc),
            nn.ReLU(inplace=True))
            for ii in (chnn_side+chnn_side)]
        self.C_cat = nn.ModuleList(C_cat)
        idx = [ii for ii in range(len(chnn_side))]
        C_up = [nn.Sequential(
            nn.Conv2d(chnn_targ[ii]+chnn_side[ii]//rd_sc, chnn_targ[ii+1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(chnn_targ[ii+1]),
            nn.ReLU(inplace=True))
            for ii in (idx+idx)]
        self.C_up = nn.ModuleList(C_up)
        self.C_cls = nn.Conv2d(chnn_targ[-1]*2, n_class, 1)

    def forward(self, inputs):
        img, depth = inputs
        cas_rgb, cas_dep = img[0], depth[0]
        nd_rgb, nd_dep, nd_key = None, None, False
        for ii in range(self.n_graph):
            feat_rgb, feat_dep = self.graph[ii]([img[ii], depth[ii], nd_rgb, nd_dep], nd_key)
            feat_rgb = torch.cat(feat_rgb, 1)
            feat_rgb = self.C_cat[ii](feat_rgb)
            feat_dep = torch.cat(feat_dep, 1)
            feat_dep = self.C_cat[self.n_graph+ii](feat_dep)
            nd_rgb, nd_dep, nd_key = feat_rgb, feat_dep, True
            cas_rgb = torch.cat((feat_rgb, cas_rgb), 1)
            cas_rgb = F.interpolate(cas_rgb, scale_factor=2, mode='bilinear', align_corners=True)
            cas_rgb = self.C_up[ii](cas_rgb)
            cas_dep = torch.cat((feat_dep, cas_dep), 1)
            cas_dep = F.interpolate(cas_dep, scale_factor=2, mode='bilinear', align_corners=True)
            cas_dep = self.C_up[self.n_graph+ii](cas_dep)
        feat = torch.cat((cas_rgb, cas_dep), 1)
        out = self.C_cls(feat)
        return out


class CasGnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_rgb = VGG()
        self.enc_dep = VGG()
        # Cascade Graph Reasoning
        self.graph = CGR()        

    def forward(self, inputs):
        img, depth = inputs
        feat_rgb = self.enc_rgb(img)
        feat_dep = self.enc_dep(depth)
        out = self.graph([feat_rgb, feat_dep])
        return out
