from IPython.core.debugger import set_trace

import torch
import torch.nn as nn

#layer_factor=16 if args.dim==2 else 128        
#Q = Q_MNIST_BN(layer_factor = 16, z_dim = 10)
#G = G_MNIST_BN(layer_factor = 16, z_dim =10)

# for energy,    
# if basic energy is not enough, consider g_depth


class G_CIFAR_BN(nn.Module):
    def __init__(self,layer_factor = 64,z_dim = 10, nc=1,w=28,device = "cuda",**kw):
        super(G_CIFAR_BN, self).__init__()
        # Define Layers
        preprocess = nn.Sequential(
            nn.Linear(z_dim, 4*4*4*layer_factor),
            nn.BatchNorm1d(4*4*4*layer_factor),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*layer_factor, 2*layer_factor, 5),
            nn.BatchNorm2d(2*layer_factor),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*layer_factor, layer_factor, 5),
            nn.BatchNorm2d(layer_factor),
            nn.ReLU(True),
        )
        block3 = nn.Sequential(
            nn.ConvTranspose2d(layer_factor,layer_factor//2, 5),
            nn.BatchNorm2d(layer_factor//2),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(layer_factor//2, nc, 2, stride=2)
        # Define Network Layers
        self.block1 = block1
        self.block2 = block2
        self.block3 = block3
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()
        self.d = layer_factor
        self.nc = nc
        self.w = w
        self.to(device)
        self.device = device
        self.train()
        
    # Define forward function
    def forward(self, z):

        #z = atanh(z)
        output = self.preprocess(z)

        output = output.view(-1, 4*self.d, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        return output.view(-1, self.nc, self.w, self.w) 

