import argparse
import glob
import os
import time

import torch
import torch.nn.functional as F
from models import *
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--dataset', type=str, default='PROTEINS', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')

# modified for adding dummy node
from add_dummy import add_dummy_PyG_TU_data
parser.add_argument('--dummy', type=bool, default=False, help='whether to add a dummy node in the graph')

# for specifying special models
parser.add_argument('--model', type=str, default='Model', help='the model structure in models.py')

# for adding trainable weight on dummy edges
parser.add_argument('--dummy_weight', type=float, default=0., help='whether or not to add additional trainable weight on dummy edges. when set to 0.0 this part is disabled.')
parser.add_argument('--mul_avg_node', type=bool, default=False, help='whether the edge weight is multiplied by the avg. number of nodes in graph')

# for adding transformation layer to dummy nodes
parser.add_argument('--transform_dummy', type=str, default='', help='[no_activation|OTHER_NON_EMPTY_STRING]')

# for additional arguments of the baseline models
parser.add_argument('--additional', type=str, default='', help='Additional parameters for the baseline models')
                    
args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    
# modified: add setting seeds for random and numpy
import random
import numpy as np
random.seed(args.seed)
np.random.seed(args.seed)

if args.dummy:
    print('Running on dataset {}, with dummy node'.format(args.dataset))
    dataset = TUDataset(os.path.join('data', args.dataset+'_dummy'), name=args.dataset, use_node_attr=True, pre_transform=add_dummy_PyG_TU_data)
    print(dataset[0])
    print(dataset[0].edge_index)
else:
    print('Running on dataset {}, without dummy node'.format(args.dataset))
    dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, use_node_attr=True)

# set maximum number of nodes, useful for some models (e.g. DiffPool)
max_num_nodes = max([item.x.size(0) for item in dataset])
setattr(dataset, 'max_num_nodes', max_num_nodes)
args.max_num_nodes = max_num_nodes
print('Maximum number of nodes:', max_num_nodes)


# Are these two still right after adding dummy node? YES
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

print(args)

num_training = int(len(dataset) * 0.8)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

if args.model == 'Model':
    model = eval(args.model+'(args).to(args.device)')
else:
    model = eval(args.model+'(args).to(args.device)')
    
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def train():
    min_loss = 1e10
    patience_cnt = 0
    val_loss_values = []
    best_epoch = 0

    t = time.time()
    model.train()
    # Collect train/val information
    info = {'loss':{'train': [], 'val': []}, 'acc':{'train': [], 'val': []}}
    for epoch in range(args.epochs):
        loss_train = 0.0
        correct = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        acc_train = correct / len(train_loader.dataset)
        acc_val, loss_val = compute_test(val_loader)
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
              'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(loss_val),
              'acc_val: {:.6f}'.format(acc_val), 'time: {:.6f}s'.format(time.time() - t))
        
        # Collect train/val information
        info['loss']['train'].append(loss_train)
        info['loss']['val'].append(loss_val)
        info['acc']['train'].append(acc_train)
        info['acc']['val'].append(acc_val)

        val_loss_values.append(loss_val)
        torch.save(model.state_dict(), '{}.pth'.format(epoch))
        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)

    files = glob.glob('*.pth')
    for f in files:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))
    
    # save other information
    torch.save([info, best_epoch, time.time()-t], 'info.pth')

    return best_epoch


def compute_test(loader):
    model.eval()
    correct = 0.0
    loss_test = 0.0
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_test += F.nll_loss(out, data.y).item()
    return correct / len(loader.dataset), loss_test


if __name__ == '__main__':
    # Model training
    best_model = train()
    # Restore best model for test set
    model.load_state_dict(torch.load('{}.pth'.format(best_model)))
    test_acc, test_loss = compute_test(test_loader)
    print('Test set results, loss = {:.6f}, accuracy = {:.6f}'.format(test_loss, test_acc))
    
    # save running info
    result = {
        'dataset': args.dataset,
        'dummy': args.dummy,
        'model': args.model,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'pooling_ratio': args.pooling_ratio,
        'dropout_ratio': args.dropout_ratio,
        'patience': args.patience,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'dummy_weight': model.dummy_weight.item() if args.dummy_weight > 0 else None,
        'transform_dummy': args.transform_dummy,
        'mul_avg_node': args.mul_avg_node
    }
    with open('result.txt', 'w') as f:
        import json
        f.write(json.dumps(result, indent=4, sort_keys=True, separators=(',', ':')))
    
    # move to folder
    if not os.path.exists('results'): os.mkdir('results')
    timing = time.gmtime()
    timing = '{}-{}-{}-{}:{}:{}'.format(timing[0], timing[1], timing[2], timing[3]+8, timing[4], timing[5])
    dirname = args.dataset+ ('_dummy' if args.dummy else '') + '_' + args.model + '_' + timing
    os.mkdir(dirname)
    os.system('mv info.pth {}\n mv result.txt {}\n cp {}.pth {}'.format(dirname, dirname, best_model, dirname))
    
    os.system('mv {} results'.format(dirname))