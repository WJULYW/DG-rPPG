# -*- coding: UTF-8 -*-
from basic_module import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import utils
from torchvision import models
import numpy as np

np.set_printoptions(threshold=np.inf)
sys.path.append('..')
args = utils.get_args()


class Discriminator(nn.Module):
    def __init__(self, max_iter, domain_num=4):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, domain_num)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            self.fc2
        )
        self.grl_layer = GRL(max_iter)

    def forward(self, feature):
        adversarial_out = self.ad_net(self.grl_layer(feature))
        return adversarial_out


class DG_rPPG(nn.Module):
    def __init__(self, ada_num=2):
        super(DG_rPPG, self).__init__()
        self.encoder = BaseNet()

        self.adaIN_layers = nn.ModuleList([ResnetAdaINBlock(256) for i in range(ada_num)])

        self.conv_final = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.gamma = nn.Linear(256, 256, bias=False)
        self.beta = nn.Linear(256, 256, bias=False)

        self.FC = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.ReLU(inplace=True)
        )
        self.ada_conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.ada_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.ada_conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.ada_conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256)
        )

        self.fc = nn.Linear(512, 1)


        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(512, 256, [2, 1], downsample=1),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(256, 128, [1, 1], downsample=1),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(128, 64, [2, 1], downsample=1),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(64, 32, [1, 1], downsample=1),
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(32, 16, [2, 1], downsample=1),
        )
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(16, 8, [1, 1], downsample=1),
        )
        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=[1, 2], stride=[1, 2]),
            BasicBlock(8, 1, [2, 1], downsample=1),
        )

    def sample_uncertainty_style(self, gamma_a, gamma_v_2, beta_a, beta_v_2, num=1, device='cpu'):

        sample = torch.normal(mean=0., std=1., size=(int(num), 1)).to(device)

        sample_gamma = gamma_a + sample * torch.sqrt(gamma_v_2)
        sample_beta = beta_a + sample * torch.sqrt(beta_v_2)

        return sample_gamma, sample_beta

    def cal_gamma_beta_param(self, x1):
        embs = self.encoder.get_rep(x1)

        x1_add = embs[1]
        x1_add = self.ada_conv1(x1_add) + embs[2]
        x1_add = self.ada_conv2(x1_add) + embs[3]
        x1_add = self.ada_conv3(x1_add) + embs[0]
        x1_add = self.ada_conv4(x1_add)

        domain_invariant = torch.nn.functional.adaptive_avg_pool2d(x1_add, 1).reshape(x1_add.shape[0], -1)

        gmp_ = self.FC(domain_invariant)
        gamma, beta = self.gamma(gmp_), self.beta(gmp_)

        return x1_add, gamma, beta

    def sample_uncertainty_feat(self, feat, feats_close, num=1, device='cpu'):
        var = torch.var(feats_close, dim=0, keepdim=True)
        sample = torch.normal(mean=0., std=1., size=(int(num), 1)).to(device)
        sample_feat = feat + sample * torch.sqrt(var)

        return sample_feat

    def adain_aggregate(self, x, gamma, beta, num=1, device='cpu'):
        gamma_v_2 = torch.var(gamma, dim=0, keepdim=True)
        beta_v_2 = torch.var(beta, dim=0, keepdim=True)

        gamma_a = torch.mean(gamma, dim=0, keepdim=True)
        beta_a = torch.mean(beta, dim=0, keepdim=True)
        gamma_aug, beta_aug = self.sample_uncertainty_style(gamma_a, gamma_v_2, beta_a, beta_v_2, num, device)
        x = x.unsqueeze(0).repeat(num, 1, 1, 1)
        for i in range(len(self.adaIN_layers)):
            fea_aug = self.adaIN_layers[i](x, gamma_aug, beta_aug)
        fea_aug = self.conv_final(fea_aug)
        fea_aug = self.avgpool(fea_aug).view(fea_aug.shape[0], -1)

        return fea_aug

    def forward(self, input, input2):

        x1, gamma1, beta1 = self.cal_gamma_beta_param(input)
        x2, gamma2, beta2 = self.cal_gamma_beta_param(input2)

        domain_invariant = self.conv_final(x1)
        domain_invariant = self.avgpool(domain_invariant).view(domain_invariant.shape[0], -1)

        fea_x1_x1 = x1
        for i in range(len(self.adaIN_layers)):
            fea_x1_x1 = self.adaIN_layers[i](fea_x1_x1, gamma1, beta1)

        fea_x1_x1 = self.conv_final(fea_x1_x1)
        em = self.avgpool(fea_x1_x1).view(fea_x1_x1.shape[0], -1)

        HR = self.fc(em)
        # For Sig
        x = self.up1(fea_x1_x1)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.up6(x)
        Sig = self.up7(x).squeeze(dim=1)
        fea_x1_x1 = em

        return HR, Sig, fea_x1_x1, x1, gamma1, beta1, domain_invariant


