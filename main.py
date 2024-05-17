import torch
import random
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric import utils
from networks import Net
import torch.nn.functional as F
import argparse
from torch.utils.data import random_split
from cycle import add_cycle_nodes
import os
import re

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--dataset', type=str, default='NCI1',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--epochs', type=int, default=100000,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--ablation', type=int, default=0)

args = parser.parse_args()
ablation = args.ablation

# device
args.device = 'cpu'
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'

# random seed
args.seed = random.randint(0, 1000)
torch.manual_seed(args.seed)

dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset)
if ablation != 2:
    dataset = add_cycle_nodes(dataset, ablation)

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

num_training = int(len(dataset)*0.8)
num_val = int(len(dataset)*0.1)
num_test = len(dataset) - (num_training+num_val)
training_set,validation_set,test_set = random_split(dataset,[num_training,num_val,num_test])


train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set,batch_size=args.batch_size,shuffle=False)
test_loader = DataLoader(test_set,batch_size=2,shuffle=False)
model = Net(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def test(model,loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out,data.y,reduction='sum').item()
    return correct / len(loader.dataset),loss / len(loader.dataset)


# Find the next log file index
def get_next_log_index():
    # if folder doesn't exist, make one
    if not os.path.exists(f'./logs/{args.dataset}/ablation_{ablation}'):
        os.makedirs(f'./logs/{args.dataset}/ablation_{ablation}')
    log_files = [f for f in os.listdir(f'./logs/{args.dataset}/ablation_{ablation}') if re.match(r'log_\d+\.txt', f)]
    if not log_files:
        return 1
    max_index = max(int(re.search(r'\d+', f).group()) for f in log_files)
    return max_index + 1 

log_index = get_next_log_index()
log_filename = f'./logs/{args.dataset}/ablation_{ablation}/log_{log_index}.txt'

# Logging function
def log(message):
    with open(log_filename, 'a') as f:
        f.write(message + '\n')

# Log training configuration
log(f'Training configuration: {args}')

min_loss = 1e10
patience = 0

for epoch in range(args.epochs):
    model.train()
    for i, data in enumerate(train_loader):
        data = data.to(args.device)
        out = model(data)
        loss = F.nll_loss(out, data.y)
        print("Training loss:{}".format(loss.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    val_acc,val_loss = test(model,val_loader)
    log(f'Epoch {epoch}: Validation loss: {val_loss}\taccuracy: {val_acc}')
    print(f'Epoch {epoch}: Validation loss: {val_loss}\taccuracy: {val_acc}')
    if val_loss < min_loss:
        torch.save(model.state_dict(),'latest.pth')
        print("Model saved at epoch{}".format(epoch))
        min_loss = val_loss
        patience = 0
    else:
        patience += 1
    if patience > args.patience:
        break 

model = Net(args).to(args.device)
model.load_state_dict(torch.load('latest.pth'))
test_acc,test_loss = test(model,test_loader)
log(f'Test accuracy: {test_acc}')
print(f'Test accuracy: {test_acc}')