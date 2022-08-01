import argparse
import glob
import os
import shutil
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
from dataset import PYGDataset
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
from models import *

# modified for adding dummy node
from utils.io import load_config, save_config, save_results
from utils.log import init_logger, close_logger, generate_log_line, generate_best_line, get_best_epochs
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score


def train(args, model, optimizer, train_loader, valid_loader, logger=None, writer=None):
    min_loss = 1e10
    patience_cnt = 0
    valid_valid_lossues = []
    best_epoch = 0
    epoch_steps = len(train_loader)

    # t = time.time()
    model.train()
    # Collect train/val information
    info = {'loss': {'train': [], 'val': []}, 'acc': {'train': [], 'val': []}}
    for epoch in range(args.epochs):
        train_loss = 0.0
        correct = 0
        for batch_id, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()

            if writer:
                step = batch_id + epoch * epoch_steps
                writer.add_scalar("train/train-nll", loss.item(), step)
                writer.add_scalar("train/eval-accuracy", pred.eq(data.y).sum().item() / data.y.shape[0], step)

        train_acc = correct / len(train_loader.dataset)
        valid_results = eval_test(valid_loader)
        valid_loss = valid_results["error"]["nll"].mean()
        valid_acc = valid_results["error"]["accuracy"]

        if logger:
            logger.info("-" * 80)
            logger.info(
                generate_log_line(
                    "train",
                    epoch=epoch,
                    total_epochs=args.epochs,
                    **{
                        "\n" + " " * (getattr(logger, "prefix_len") + 1) + "train-loss": "{:6.3f}".format(train_loss),
                        "eval-accuracy": "{:.6f}".format(train_acc)
                    }
                )
            )
            logger.info(
                generate_log_line(
                    "valid",
                    epoch=epoch,
                    total_epochs=args.epochs,
                    **{
                        "\n" + " " * (getattr(logger, "prefix_len") + 1) + "valid-loss": "{:6.3f}".format(valid_loss),
                        "eval-accuracy": "{:.6f}".format(valid_acc)
                    }
                )
            )

        if writer:
            writer.add_scalar("train/train-nll-epoch", train_loss, epoch)
            writer.add_scalar("train/eval-accuracy-epoch", train_acc, epoch)
            writer.add_scalar("valid/train-nll-epoch", valid_loss, epoch)
            writer.add_scalar("valid/eval-accuracy-epoch", valid_acc, epoch)

        valid_valid_lossues.append(valid_loss)
        torch.save(model.state_dict(), os.path.join(args.save_model_dir, '{}.pt'.format(epoch)))

        if valid_valid_lossues[-1] < min_loss:
            min_loss = valid_valid_lossues[-1]
            best_epoch = epoch
            patience_cnt = 0
            shutil.copyfile(
                os.path.join(args.save_model_dir, '{}.pt'.format(epoch)),
                os.path.join(args.save_model_dir, 'best.pt')
            )
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

    return best_epoch


def eval_test(loader):
    model.eval()
    results = {
        "data": {
            "id": list(),
            "label": list()
        },
        "prediction": {
            "pred": list()
        },
        "error": {
            "nll": list(),
            "accuracy": 0.0,
            "f1": 0.0
        },
        "time": {
            "avg": list(),
            "total": 0.0
        }
    }
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        loss = F.nll_loss(out, data.y, reduction="none")

        results["data"]["label"].append(data.y.cpu().view(-1))
        results["prediction"]["pred"].append(pred.detach().cpu().view(-1))
        results["error"]["nll"].append(loss.detach().cpu().view(-1))
    results["data"]["label"] = torch.cat(results["data"]["label"], dim=0)
    results["prediction"]["pred"] = torch.cat(results["prediction"]["pred"], dim=0)
    results["error"]["nll"] = torch.cat(results["error"]["nll"], dim=0)

    label = results["data"]["label"].numpy()
    pred = results["prediction"]["pred"].numpy()
    results["error"]["accuracy"] = accuracy_score(label, pred)
    results["error"]["f1"] = f1_score(label, pred, average="macro")
    return results


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()

    # for data config
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default="../data_processing/tu_data",
        help="the directory of gram matrices"
    )
    parser.add_argument('--dataset', type=str, default='PROTEINS', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES')
    parser.add_argument('--add_dummy', type=str, default="false", help='whether to add a dummy node in the graph (true/false)')
    parser.add_argument('--convert_conjugate', type=str, default="false", help='whether to convert to conjugate graphs (true/false)')

    # for specifying special models
    parser.add_argument('--model', type=str, default='GCN', help='the model structure in models.py')

    parser.add_argument("--save_model_dir", type=str, default="checkpoints")

    # for model config
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden size')
    parser.add_argument('--sample_neighbor', type=str, default="true", help='whether sample neighbors (true/false)')
    parser.add_argument('--sparse_attention', type=str, default="true", help='whether use sparse attention (true/false)')
    parser.add_argument('--structure_learning', type=str, default="true", help='whether perform structure learning (true/false)')
    parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
    parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
    parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
    parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
    parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')

    # for adding trainable weight on dummy edges
    parser.add_argument(
        '--dummy_weight',
        type=float,
        default=0.1,
        help='whether or not to add additional trainable weight on dummy edges. when set to 0.0 this part is disabled.'
    )
    parser.add_argument(
        '--mul_avg_node',
        type=bool,
        default=False,
        help='whether the edge weight is multiplied by the avg. number of nodes in graph'
    )

    # for adding transformation layer to dummy nodes
    parser.add_argument('--transform_dummy', type=str, default='', help='[no_activation|OTHER_NON_EMPTY_STRING]')

    # for additional arguments of the baseline models
    import json
    parser.add_argument('--additional', type=json.loads, default='{}', help='Additional parameters for the baseline models')
                        
    args = parser.parse_args()

    os.makedirs(args.save_model_dir, exist_ok=True)
    logger = init_logger(log_file=os.path.join(args.save_model_dir, "log.txt"), log_tag=args.model)
    writer = SummaryWriter(args.save_model_dir)
    save_config(args, os.path.join(args.save_model_dir, "config.json"))

    # modified: add setting seeds for random and numpy
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info('Running on dataset {}'.format(args.dataset))
    dataset = PYGDataset(
        args.dataset_dir, name=args.dataset, use_node_attr=True,
        add_dummy=args.add_dummy == "true", convert_conjugate=args.convert_conjugate == "true"
    )

    # set maximum number of nodes, useful for some models (e.g. DiffPool)
    max_num_nodes = max([item.x.size(0) for item in dataset])
    setattr(dataset, 'max_num_nodes', max_num_nodes)
    args.max_num_nodes = max_num_nodes
    logger.info('Maximum number of nodes: {}'.format(max_num_nodes))

    # set number of relations for Relational conv models
    args.num_relations = dataset.num_edge_features
    logger.info('Dataset # edge features: {}'.format(args.num_relations))

    # Are these two still right after adding dummy node? YES
    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features

    num_training = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_training + num_val)
    train_set, valid_set, test_set = random_split(dataset, [num_training, num_val, num_test])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    if args.model == 'Model':
        model = eval(args.model + '(args).to(args.device)')
    else:
        model = eval(args.model + '(args).to(args.device)')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Model training
    best_epoch = train(args, model, optimizer, train_loader, valid_loader, logger, writer)
    # Restore best model for test set
    model.load_state_dict(torch.load(os.path.join(args.save_model_dir, 'best.pt')))

    test_results = eval_test(test_loader)
    test_loss = test_results["error"]["nll"].mean()
    test_acc = test_results["error"]["accuracy"]
    logger.info(
        generate_log_line(
            "test",
            epoch=-1,
            total_epochs=args.epochs,
            **{
                "\n" + " " * (getattr(logger, "prefix_len") + 1) + "valid-loss": "{:6.3f}".format(test_loss),
                "eval-accuracy": "{:.6f}".format(test_acc)
            }
        )
    )
