import torch
import torch.nn as nn
import torchvision

class Generator(torch.nn.Module):
    def __init__(self,ngpu):
        super(Generator,self).__init__()
        self.ngpu = ngpu

        self.main = torch.nn.Sequential(
            #inpput is Z , going into covolution
            nn.ConvTranspose2d(in_channels=nz,out_channels=ngf*8,kernel_size=4,stride=1,padding = 0),
            nn.BatchNorm2d(num_features=num_ngf*8),
            nn.ReLU(inplace = True),

            #state size (ngf*8)x 4x 4
            nn.ConvTranspose2d(ngf*8,ngf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            #state size (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            #state size (ngf*2)x16x16
            nn.ConvTranspose2d(ngf*2,ngf,4,2,1,bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)

            #state size (ngf)x32x32
            nn.ConvTranspose2d(ngf)
            nn.Tanh()

            #state size (nc)x64x64
        )

    def forward(self,input):
        return self.main(input)
