import torch
from torch import nn, optim
import torch.functional as F
from torch.nn.parameter import Parameter
from torch.nn.functional import normalize
from utils.utils import build_affinity_matrix, sgc_precompute
from typing import Optional

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.PReLU(),
            nn.Linear(500, 500),
            nn.PReLU(),
            nn.Linear(500, 2000),
            nn.PReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)
        

class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.PReLU(),
            nn.Linear(2000, 500),
            nn.PReLU(),
            nn.Linear(500, 500),
            nn.PReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        xr = self.decoder(x)
        return xr

class ClusteringLayer(nn.Module):
    def __init__(self, class_num, hidden_dimension, alpha: float = 1.0, cluster_centers: Optional[torch.Tensor] = None):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.

        :param class_num: number of clusters
        :param hidden_dimension: hidden dimension, output of the encoder
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super(ClusteringLayer, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.class_num = class_num
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.class_num, self.hidden_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)


    def forward(self, batch):
        """
        :param batch: [batch size, embedding dimension] FloatTensor
        :return: [batch size, number of clusters] FloatTensor
        """
        # print('self.cluster_centers', self.cluster_centers.shape)
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, in_fea, out_fea):
        super(SGC, self).__init__()

        self.W = nn.Sequential(nn.Linear(in_fea, out_fea), nn.PReLU())

    def forward(self, x):
        x = self.W(x)
        return x


class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, out_feature, class_num, neighbor_num, device):
        super(Network, self).__init__()
        self.GCNencoders = []
        self.encoders = []
        self.decoders = []
        self.decs = []
        self.embed_projs = []
        self.clusters = []
        self.cluster_projs = []
        self.view = view
        self.class_num = class_num
        self.device = device 
        self.neighbor_num = neighbor_num
        self.alpha = 1.0
        # input_size: encoder input dim
        # feature_dim: encoder output dim, gcn input dim
        # out_feature: gcn output dim
        for v in range(view):
            encoder = Encoder(input_size[v], feature_dim)
            decoder = Decoder(input_size[v], feature_dim)
            self.encoders.append(encoder.to(device))
            self.decoders.append(decoder.to(device))
            self.embed_projs.append(nn.Sequential(nn.Linear(feature_dim, out_feature)))
            self.GCNencoders.append(SGC(out_feature, out_feature))


        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.GCNencoders = nn.ModuleList(self.GCNencoders)
        self.embed_projs = nn.ModuleList(self.embed_projs)
        

        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=out_feature*view, nhead=1, dim_feedforward=256)
        self.TransformerEncoder = nn.TransformerEncoder(self.TransformerEncoderLayer, num_layers=1)
        self.embed_proj = nn.Sequential(nn.Linear(feature_dim, out_feature))
        self.fusion_proj = nn.Sequential(nn.Linear(out_feature*view, out_feature))
        self.GCNencoder = SGC(out_feature, out_feature)
        self.cluster_proj = nn.Sequential(
                                    nn.Linear(out_feature, out_feature),
                                    nn.PReLU(),
                                    nn.Linear(out_feature, class_num),
                                    nn.Softmax(dim=1)
                                    )
        

    def forward(self, xs):
        hs = []
        zs = []
        xrs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            f = self.embed_proj(z)
            hs.append(f)
            zs.append(z)
            xrs.append(xr)
        return xrs, hs, zs
    

    def forward_fusion(self, xs):
        hs = []
        qs = []
        tars = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            f = self.embed_proj(z)
            adj = build_affinity_matrix(f, self.neighbor_num).to(self.device)
            features, precompute_time = sgc_precompute(f, adj)
            h_ = self.GCNencoders[v](features)
            h = normalize(h_, dim=1)
            q = self.cluster_proj(h)
            hs.append(h)
            qs.append(q)

        cat_feautre = torch.cat(hs, dim=1)
        cat_feautre = torch.unsqueeze(cat_feautre, dim=1)
        fusion_fea = self.TransformerEncoderLayer(cat_feautre)
        fusion_fea = torch.squeeze(fusion_fea, dim=1)
        norm_fusion_fea = normalize(self.fusion_proj(fusion_fea), dim=1)
        p = self.cluster_proj(norm_fusion_fea)
        tp = []

        return  hs, norm_fusion_fea, qs, tars, p, tp
    
