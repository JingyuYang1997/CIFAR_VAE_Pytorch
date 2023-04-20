import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
from ipdb import set_trace as debug

class CIFAREncoder(nn.Module):
    def __init__(self,rep_dim):
        super(CIFAREncoder,self).__init__()
        self.rep_dim = rep_dim

        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Sequential(
            nn.Linear(128 * 4 * 4, self.rep_dim))

        self.fc2 = nn.Sequential(
            nn.Linear(128 * 4 * 4, self.rep_dim))

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool(F.relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.relu(self.bn2d3(x)))
        x = x.view(-1,128*4*4)
        mu = self.fc1(x)
        logvar = self.fc2(x)
        return mu, logvar


class CIFARDecoder(nn.Module):
    def __init__(self,rep_dim):
        super(CIFARDecoder,self).__init__()
        self.rep_dim = rep_dim

        self.fc = nn.Linear(rep_dim, 128*4*4)
        self.deconv1 = nn.ConvTranspose2d(128, 128, 5, padding=2)
        self.bn2d1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, padding=2)
        self.bn2d3 = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, padding=2)

    def forward(self,x):
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)
        # debug()
        x = self.deconv1(x)
        x = F.interpolate(F.relu(self.bn2d1(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.relu(self.bn2d2(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.relu(self.bn2d3(x)), scale_factor=2)
        x = self.deconv4(x)
        # x = torch.sigmoid(x)
        return x


