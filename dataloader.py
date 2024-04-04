from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch



class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()



class LabelMe(Dataset):
    def __init__(self, path):
        self.view1 = scipy.io.loadmat(path+'LabelMe.mat')['X1'].astype(np.float32)
        self.view2 = scipy.io.loadmat(path+'LabelMe.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'LabelMe.mat')['Y']#.transpose()
        self.y = labels
    def __len__(self):
        return 2688
    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()







def load_data(dataset):
    if dataset == "Fashion":
        dataset = Fashion('./data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "LabelMe":
            dataset = LabelMe('./data/')
            dims = [512, 245]
            view = 2
            data_size = 2688
            class_num = 8


    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
