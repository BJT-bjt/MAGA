import torch
from torch import nn
from SSGCNetwork import Network
from metric import valid, evaluate
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
import os
import warnings
from sklearn.cluster import KMeans
from utils.utils import neraest_labels
import time

st = time.time()

torch.set_num_threads(4)
# MNIST-USPS
# BDGP
# LableMe
# Fashion
Dataname = 'Fashion'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument('--neighbor_num', default=5, type=int)
parser.add_argument('--feature_dim', default = 512, type=int)
parser.add_argument('--gcn_dim', default = 128, type=int)
parser.add_argument('--tau', default= 0.1 , type=float)
parser.add_argument('--lambda1', default = 1.0 , type=float)
parser.add_argument('--lambda2', default = 1.0 , type=float)
parser.add_argument('--eta', default = 1.0, type=float)
parser.add_argument('--neg_size', default = 128, type=int)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings('ignore')


if args.dataset == "Fashion":
    args.mse_epochs = 200
    args.con_epochs = 200
    seed = 10
if args.dataset == 'LabelMe':
    args.mse_epochs = 200
    args.con_epochs = 200
    seed = 10

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(seed)

dataset, dims, view, data_size, class_num = load_data(args.dataset)


data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

# 添加噪声函数
def add_noise(input, noise_factor=0.1):
    noise = torch.randn_like(input) * noise_factor
    return input + noise


def pretrain(epoch):
    tot_loss = 0.
    MSE = nn.MSELoss()
    xns=[]
    for batch_idx, (xs, labels, _) in enumerate(data_loader):
        for v in range(view):
            xn = add_noise(xs[v])
            xns.append(xn)
            xns[v] = xns[v].to(device)
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, hs, f = model(xns)
        loss_list = []
        for v in range(view):
            loss_list.append(F.mse_loss(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
    return hs, labels, hs




def model_train(epoch):
    total_loss = 0.

    for batch_idx, (xs, labels, _) in enumerate(data_loader):
        xns=[]
        for v in range(view):
            xn = add_noise(xs[v])
            xns.append(xn)
            xns[v] = xns[v].to(device)
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, hs, zs = model(xns)
        hs, f, qs, tars, p, tp = model.forward_fusion(xs)

        kmeans = KMeans(n_clusters=class_num, n_init=10) 
        y_pred_km = kmeans.fit_predict(f.data.cpu().numpy())
        y_pred_km = y_pred_km.flatten()
        y_pred = y_pred_km

        loss_list = []
        for v in range(view):
            loss_list.append(F.mse_loss(xs[v], xrs[v])) # reconstruction loss
            loss_list.append(criterion.forword_debiased_instance(f, hs[v], y_pred))
            
            loss_list.append(criterion.forword_feature(f.T, hs[v].T))

            qn = neraest_labels(f, qs[v]).to(device)
            loss_list.append(criterion.forward_pui_label(p, qn))
        
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(total_loss/len(data_loader)))  
    return f, labels



accs = []
nmis = []
purs = []
aris = []

if not os.path.exists('./models'):
    os.makedirs('./models')

T=1
for i in range(T):
    print("ROUND:{}".format(i+1))



    #Network train
    model = Network(view, dims, args.feature_dim, args.gcn_dim, class_num, args.neighbor_num,  device)
    # print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)
    criterion  = Loss(args.batch_size, class_num, args.temperature_f, args.lambda1, args.lambda2, args.eta, args.neg_size, device).to(device)

    epoch = 1
    # pretrain
    while epoch <= args.mse_epochs:
        zs, labels, hs = pretrain(epoch)
        epoch += 1
        
    hc = torch.cat(hs, dim=1)
    kmeans = KMeans(n_clusters=class_num, n_init=10) 
    y_pred = kmeans.fit_predict(hc.data.cpu().numpy())
    y_pred = y_pred.flatten()
    labels = labels.flatten()
    labels = labels.data.cpu().numpy()
    nmi, ari, acc, pur = evaluate(labels, y_pred)
    print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))
        

    while epoch <= args.mse_epochs + args.con_epochs:
        fusion, labels = model_train(epoch)

        if epoch % 10 == 0:
            acc, nmi, pur, ari, acc_k, nmi_k, ari_k,  pur_k= valid(model, device, dataset, view, data_size, class_num, eval_h=False)
            print("Clustering results on semantic labels: ")
            print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))
            print("Clustering results on kmeans clustering: ")
            print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc_k, nmi_k, ari_k, pur_k))

        epoch += 1


    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

        
   

