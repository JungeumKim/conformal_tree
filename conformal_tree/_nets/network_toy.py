import torch


class MLP(torch.nn.Module):
    def __init__(self, device="cuda", input_dim=2, out_dim=1, leaky=0.1, factor = 32,cache=None):
        super().__init__()
        self.input_dim =input_dim
        self.non_linear = torch.nn.LeakyReLU(leaky) if leaky>0 else torch.nn.ReLU()
        # TODO: init - it may affect the results.

        self.l1 = torch.nn.Linear(input_dim, factor)
        self.l2 = torch.nn.Linear(factor, factor)
        self.l3 = torch.nn.Linear(factor, factor)
        self.l4 = torch.nn.Linear(factor, out_dim)

        self.to(device)
        self.device = device
        
        self.cache = cache

    def forward(self, x):

        h = self.non_linear(self.l1(x))
        h = self.non_linear(self.l2(h))
        h = self.non_linear(self.l3(h))
        h = self.l4(h)
        return h