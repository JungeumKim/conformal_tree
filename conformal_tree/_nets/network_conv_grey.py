from IPython.core.debugger import set_trace

import torch
import torch.nn as nn

#layer_factor=16 if args.dim==2 else 128        
#Q = Q_MNIST_BN(layer_factor = 16, z_dim = 10)
#G = G_MNIST_BN(layer_factor = 16, z_dim =10)

# for energy,    
# if basic energy is not enough, consider g_depth


class Q_MNIST_BN(nn.Module):
    def __init__(self,layer_factor = 64,z_dim = 8, nc = 1, w=28, device = "cuda",**kw):
        super(Q_MNIST_BN, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(nc, layer_factor, 5, stride=2, padding=2),
            nn.BatchNorm2d(layer_factor),
            nn.ReLU(True),
            nn.Conv2d(layer_factor, 2*layer_factor, 5, stride=2, padding=2),
            nn.BatchNorm2d(2*layer_factor),
            nn.ReLU(True),
            nn.Conv2d(2*layer_factor, 4*layer_factor, 5, stride=2, padding=2),
            nn.BatchNorm2d(4*layer_factor),
            nn.ReLU(True),
        )
        #self.tanh = nn.Tanh()
        self.main = main
        self.output = nn.Linear(4*4*4*layer_factor, z_dim)
        self.d = layer_factor
        self.nc = nc
        self.w = w
        self.to(device)
        self.device =device
    def forward(self, input):
        #print("Q!")
        input = input.view(-1, self.nc, self.w, self.w)
        out = self.main(input)
        out = out.view(-1, 4*4*4*self.d)
        out = self.output(out)
        #out = self.tanh(out)
        return out
    
class G_MNIST_BN(nn.Module):
    def __init__(self,layer_factor = 64,z_dim = 10, nc=1,w=28, device = "cuda",input_noise=0, latent_noise = 0,**kw):
        super(G_MNIST_BN, self).__init__()
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
        deconv_out = nn.ConvTranspose2d(layer_factor, nc, 8, stride=2)
        # Define Network Layers
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()
        self.d = layer_factor
        self.nc = nc
        self.w = w
        self.to(device)
        self.device =device
        self.train()
        self.input_noise = input_noise 
        self.latent_noise = latent_noise
        
    # Define forward function
    def forward(self, z):
        #print("G!")
        #z = atanh(z)
        
        noise = torch.zeros_like(z)
        z_noisy = z + noise.normal_(0, 1) * self.latent_noise
        
        output = self.preprocess(z)

        output = output.view(-1, 4*self.d, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.deconv_out(output)
    
        output = self.sigmoid(output)
        
        noise = torch.zeros_like(output)
        output = (output+noise.normal_(0, 1)*self.input_noise).clip(0,1)
        
        return output.view(-1, self.w*self.w)#.view(-1, self.nc, self.w, self.w) 

   