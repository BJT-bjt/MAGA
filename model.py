import torch
import torch.nn as nn
from fusion_model  import FeatureFusion as fusioner
from torch.nn.parameter import Parameter


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)
        

class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512), 
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forword(self, x):
        xr = self.decoder(x)
        return xr

class SelfExpression(nn.Module):
    def __init__(self, input_size, view):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(input_size, input_size, dtype=torch.float32), requires_grad=True)

    def forword(self, x, view):
        Coef = self.Coefficient
        recon = torch.matmul(Coef, x)
        return recon, Coef


class SubspaceBase(nn.Module):
    def __init__(self, input_size, class_num):
        super(SubspaceBase, self).__init__()
        # Subspace bases proxy
        self.Dx = Parameter(torch.Tensor(input_size, class_num))

    def forward(self, x):
        Dx = self.Dx

        return Dx

class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, class_num,  d, eta, device):
        super(Network, self).__init__()
        self.encoder = []
        self.decoder = []
        self.selfexpression = []
        self.subspacebase = []
        self.view = view
        self.class_num = class_num
        self.feature_dim = feature_dim
        self.input_size = input_size
        self.d = d
        self.eta = eta

        for v in range(view):
            self.encoder.append(Encoder(self.input_size[v], self.feature_dim).to(device))
            self.decoder.append(Decoder(self.input_size[v], self.feature_dim).to(device))
            # self.selfexpression.append(SelfExpression(input_size[v], view).to(device))
            self.subspacebase.append(SubspaceBase(self.input_size[v], self.class_num).to(device))

        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)
        # self.selfexpression = nn.ModuleList(self.selfexpression)
        # self.subspacebase = nn.ModuleList(self.subspacebase)        #这句是否是必要的？

        self.totalBase = SubspaceBase(self.input_size[v], self.class_num).to(device)


    def forword_feature(self, xs):
        xrs = []
        zs = []
        zrs = []
        Dxs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoder[v](x)
            Dx = self.subspacebase[v](x)
            xr = self.decoder[v](z)
            zs.append(z)
            Dxs.append(Dx)
            xrs.append(xr)
        return zs, Dxs,  xrs
    
    def forward_fusion(self, xs):
        zs = []
        ave = 'ave'
        for v in range(self.view):
            x = xs[v]
            z = self.encoder[v](x)
            zs.append(z)
        f = fusioner(self.view)
        cat_fea, ave_fea = f.forward(zs)
        if ave is not None:
            feaDx = self.subspacebase(ave_fea)
        else:
            feaDx = self.subspacebase(cat_fea)
        return cat_fea, ave_fea, feaDx
    
    def subspace_affinity(self, z):
        d = self.d
        eta = self.eta
        for i in range(self.class_num):
            si = torch.sum(torch.pow(torch.mm(z, self.totalBase[:,i*d:(i+1)*d]),2),1,keepdim=True)  #这里的self.Dx应当如何调用？
            if sz is None:
                sz = si
            else:
                sz =torch.cat((sz, si), 1)
        sz = (sz+eta*d)/((eta+1)*d)
        sz = (sz.t() / torch.sum(sz, 1)).t()
        return sz

    










