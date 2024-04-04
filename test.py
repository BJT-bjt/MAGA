import torch
from SSGCNetwork import Network
from metric import valid
import argparse
from dataloader import load_data

# MNIST-USPS
# BDGP
# LableMe
# Fashion
Dataname = 'BDGP'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument('--neighbor_num', default=5, type=int)
parser.add_argument('--feature_dim', default = 512, type=int)
parser.add_argument('--gcn_dim', default = 128, type=int)
parser.add_argument('--tau', default= 0.1 , type=float)
parser.add_argument('--lambda1', default = 0.5 , type=float)
parser.add_argument('--lambda2', default = 0.5 , type=float)
parser.add_argument('--eta', default = 1.0, type=float)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset, dims, view, data_size, class_num = load_data(args.dataset)
model = Network(view, dims, args.feature_dim, args.gcn_dim, class_num, args.neighbor_num,  device)
model = model.to(device)
checkpoint = torch.load('./models/' + args.dataset + '.pth')
model.load_state_dict(checkpoint)
print("Dataset:{}".format(args.dataset))
print("Datasize:" + str(data_size))
print("Loading models...")
valid(model, device, dataset, view, data_size, class_num, eval_h=False)
