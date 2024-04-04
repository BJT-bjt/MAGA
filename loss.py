import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.functional import normalize
import sys
import random
# from network import SubspaceBase

class Loss(nn.Module):
    def __init__(self, batch_size, class_num,  temperature_f, lambda1,  lambda2,  eta, neg_size, device):  
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.device = device
        self.temperature_f = temperature_f
        
        self.softmax = nn.Softmax(dim=1)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eta = eta
        self.neg_size = neg_size

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")


    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask



    def forword_feature(self, h_i, h_j):
        feature_size, _ = h_i.shape
        N = 2 * feature_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T)/self.temperature_f
        sim_i_j = torch.diag(sim, feature_size)
        sim_j_i = torch.diag(sim, -feature_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N,-1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss_contrast = self.criterion(logits, labels)
        loss_contrast /= N 
        
        return self.lambda1 * loss_contrast


    def forward_pui_label(self, ologits, plogits):
        """Partition Uncertainty Index

        Arguments:
            ologits {Tensor} -- [assignment probabilities of original inputs (N x K)]
            plogits {Tensor} -- [assignment probabilities of perturbed inputs (N x k)]

        Returns:
            [Tensor] -- [Loss value]
        """
        assert ologits.shape == plogits.shape, ('Inputs are required to have same shape')

        # ologits = self.softmax(ologits)
        # plogits = self.softmax(plogits)

        # one-hot
        similarity = torch.mm(normalize(ologits.t(), p=2, dim=1), F.normalize(plogits, p=2, dim=0))
        loss_ce = self.criterion(similarity, torch.arange(similarity.size(0)).cuda())

        # balance regularisation
        o = ologits.sum(0).view(-1)
        o /= o.sum()
        loss_ne = math.log(o.size(0)) + (o * o.log()).sum()

        loss = self.lambda2 * loss_ce + self.eta * loss_ne

        return loss
    




    def forword_debiased_instance(self, h, h_i, y_pred):

        sample_size = self.batch_size
        temperature = 0.5
        y_sam = torch.LongTensor(y_pred)
        neg_size = self.neg_size
        class_sam = []
        for m in range(np.max(y_pred) + 1):
            class_del = torch.ones(int(sample_size), dtype=bool)
            class_del[np.where(y_sam.cpu() == m)] = 0
            class_neg = torch.arange(sample_size).masked_select(class_del)
            neg_sam_id = random.sample(range(0, class_neg.shape[0]), int(neg_size))
            class_sam.append(class_neg[neg_sam_id])

        out = h
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        neg_samp = torch.zeros(neg.shape[0], int(neg_size))
        for n in range(np.max(y_pred) + 1):
            neg_samp[np.where(y_sam.cpu() == n)] = neg.cpu().index_select(1, class_sam[n])[np.where(y_sam.cpu() == n)]
        neg_samp = neg_samp.cuda()
        Ng = neg_samp.sum(dim=-1)


        out = h
        pos = torch.exp(torch.mm(out, out.t().contiguous()))
        pos = torch.diag(torch.exp(torch.mm(out, h_i.t().contiguous())))
        loss = (- torch.log(pos / (Ng))).mean()#pos + 
        return self.lambda1 * loss