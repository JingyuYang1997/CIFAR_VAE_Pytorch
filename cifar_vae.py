import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
from tqdm import tqdm
from cifar_models import CIFAREncoder, CIFARDecoder
from PIL import Image
from load_data import CIFAR10
from torch.utils.data import DataLoader
import torch.nn.functional as F


class CIFARVAE(nn.Module):
    def __init__(self,rep_dim=128):
        super(CIFARVAE,self).__init__()
        self.rep_dim = rep_dim
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.enc = CIFAREncoder(rep_dim=rep_dim)
        self.dec = CIFARDecoder(rep_dim=rep_dim)

    def reparam(self,mu, logvar):
        eps = torch.randn(mu.shape).to(self.device)
        std = torch.exp(logvar)**0.5
        z = mu +eps*std
        return z

    def forward(self,x, noise=True):
        mu, logvar = self.enc(x)
        if noise:
            z = self.reparam(mu,logvar)
        else:
            z = mu
        x_hat = self.dec(z)
        return x_hat, mu, logvar


class Solver():
    def __init__(self, rep_dim):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.vae = CIFARVAE(rep_dim=rep_dim).to(self.device)

    def kl_divergence(self, mu, logvar):
        kld = -0.5*torch.sum(1+logvar-mu**2-torch.exp(logvar), dim=-1)
        return kld.mean()

    def reconstruction_loss(self,x,x_hat):
        # recons_loss = F.l1_loss(x,x_hat)
        loss = nn.MSELoss(size_average=False)
        recons_loss = loss(x_hat,x)/x_hat.shape[0]
        return recons_loss

    def train(self,trainloader, lr, epochs, beta=1.0):
        samples = trainloader.dataset.data[0:20]
        optimizer = torch.optim.Adam(self.vae.parameters(),lr=lr)
        for epoch in range(epochs):
            self.vae.train()
            batch_kld = 0.
            batch_recons = 0.
            batch_loss = 0
            n_batches = 0
            for x,_ in tqdm(trainloader):
                x = x.to(self.device)
                x_hat, mu, logvar = self.vae(x)
                kld = self.kl_divergence(mu,logvar)
                recons_loss = self.reconstruction_loss(x,x_hat)
                loss = recons_loss + beta*kld
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_kld += kld.item()
                batch_recons += recons_loss.item()
                batch_loss += loss.item()
                n_batches +=1
            print("Epoch {}/{}\tloss: {:.8f}\trecons:{:.8f}\tkld:{:.8f}".format(epoch+1, epochs, batch_loss/n_batches,
                                                                         batch_recons/n_batches, batch_kld/n_batches))
            self.vae.eval()
            x = torch.FloatTensor(samples.transpose(0,3,1,2)).to(self.device)
            x_hat, _, _ = self.vae(x,noise=True)
            samples_recons = x_hat.permute(0,2,3,1).detach().cpu().data.numpy()
            n, h,w,c = samples.shape
            img = np.concatenate([samples.reshape(n*h,w,c),samples_recons.reshape(n*h,w,c)],axis=1)
            img = Image.fromarray((img*255).astype(np.uint8))
            if not os.path.exists('./results'):
                os.mkdir('./results')
            img.save('./results/test_epoch{}.png'.format(epoch+1))

if __name__=='__main__':
    trainset = CIFAR10(normal_class=6,sup=False)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    solver = Solver(rep_dim=256)
    solver.train(trainloader,lr=0.0001,epochs=1000,beta=0.01)




