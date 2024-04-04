import torch
import faiss
import numpy as np
from collections import defaultdict
from time import perf_counter
import scipy.sparse as sp

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from torch.nn.functional import normalize



def build_affinity_matrix(X, k):
    # 将数据集转换为Tensor对象
    X = X.clone().detach()#torch.tensor(X).float()
    X = X.cpu().numpy()

    # 初始化IndexFlatL2对象
    index = faiss.IndexFlatL2(X.shape[1])
    # 将数据集加入到索引中
    index.add(X)
    # 利用索引查找每个向量的k个最近邻点
    _, ind = index.search(X, k+1)
    
    # 计算每个向量与其k个最近邻点之间的距离
    dist = np.array([np.linalg.norm(X[i]-X[ind[i][1:]], axis=1) for i in range(X.shape[0])])
    dist = torch.tensor(dist)
    # dist = torch.norm(X[:, None, :] - X[ind[:, 1:]], dim=2)
    # 将距离转换为亲和值
    aff = torch.exp(-dist ** 2 / 2)
    # 构造亲和矩阵
    W = torch.zeros(X.shape[0], X.shape[0])

    
    for i in range(X.shape[0]):
        W[i, ind[i][1:]] = aff[i]
        W[ind[i][1:], i] = aff[i]
    adj = np.array(W)
    normalization = 'NormAdj'
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    # adj = adj.astype("float")# torch.float(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    ind = torch.from_numpy(ind)
    return adj#, ind





def sgc_precompute(features, adj, degree=16, alpha=0.9):
    t = perf_counter()
    ori_features = features
    emb = alpha * features
    for i in range(degree):
        features = torch.spmm(adj, features)
        emb = emb + (1-alpha)*features/degree
    precompute_time = perf_counter()-t
    return emb, precompute_time


def normalize_embeddings(a, eps=1e-8):
    a_n = a.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    return a_norm


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a = normalize_embeddings(a, eps)
    b = normalize_embeddings(b, eps)

    sim_mt = torch.mm(a, b.transpose(0, 1))
    return sim_mt


def normalized_adjacency(adj):
   adj = adj
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def fetch_normalization(type):
   switcher = {
       'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
       'NormAdj': normalized_adjacency,  # A' = (D)^-1/2 * ( A) * (D)^-1/2
   }
   func = switcher.get(type, lambda: "Invalid normalization technique.")
   return func

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def neraest_labels(instance, labels):
    instance = instance.clone().detach()#torch.tensor(instance).float()
    instance = instance.cpu().numpy()
    labels = labels.clone().detach()#torch.tensor(labels).float()
    labels = labels.cpu().numpy()
    # 创建 Faiss 索引
    index = faiss.IndexFlatL2(instance.shape[1])  # 使用 FlatL2 索引
    # 将instance的数据添加到索引中
    index.add(instance)
    # 搜索最近邻
    k = 1
    distances, indices = index.search(instance, k)
    # print('indices', indices.flatten())
    # 获取最近邻样本的标签
    topK_labels = labels[indices]
    topK_labels =torch.from_numpy(topK_labels)
    topK_labels = torch.squeeze(topK_labels, dim=1)
    return torch.FloatTensor(topK_labels)

def knn_retrieve(V, T):
    V = V.clone().detach()#torch.tensor(instance).float()
    V = V.cpu().numpy()
    T = T.clone().detach()#torch.tensor(labels).float()
    T = T.cpu().numpy()    
    # 计算V模态样本之间的距离
    nbrs = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    nbrs.fit(V, np.arange(V.shape[0]))

    # 在V模态中检索每个样本的最近邻
    distances, indices = nbrs.kneighbors(V)

    # 根据最近邻的索引获得对应的T模态标签
    T_labels = T[indices]
    T_labels =torch.from_numpy(T_labels)
    T_labels = torch.squeeze(T_labels, dim=1)
    return T_labels

def cluster_pred(x, class_num):
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    g = x.cpu().detach().numpy()
    kmeans.fit_predict(g)
    labels = kmeans.labels_
    n_samples = len(labels)
    onehot_labels = np.zeros((n_samples, class_num))
    onehot_labels[np.arange(n_samples), labels] = 1

    predicted = torch.tensor(onehot_labels).to(torch.float32)
    return predicted

def compute_centers(x, psedo_labels):
    n_samples = x.size(0)
    class_num = psedo_labels.size(1)
    if len(psedo_labels.size()) > 1:
        weight = psedo_labels.T
    else:
        weight = torch.zeros(class_num, n_samples).to(x)  # L, N
        weight[psedo_labels, torch.arange(n_samples)] = 1
    weight = normalize(weight, p=1, dim=1)  # l1 normalization
    centers = torch.mm(weight, x)
    centers = normalize(centers, dim=1)
    return centers