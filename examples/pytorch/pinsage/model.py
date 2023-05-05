import argparse
import os
import pickle

import dgl
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import evaluation
import layers
import sampler as sampler_module


class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, textsets, hidden_dims, n_layers):
        super().__init__()

        self.proj = layers.LinearProjector(
            full_graph, ntype, textsets, hidden_dims
        )
        self.sage = layers.SAGENet(hidden_dims, n_layers)
        self.scorer = layers.ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return h_item_dst + self.sage(blocks, h_item)


def train(dataset, args):
    g = dataset["train-graph"]
    # val_matrix = dataset["val-matrix"].tocsr()
    # test_matrix = dataset["test-matrix"].tocsr()
    item_texts = dataset["item-texts"]
    user_ntype = dataset["user-type"]

    # "movie"
    item_ntype = dataset["item-type"]
    # user_to_item_etype = dataset["user-to-item-type"]
    # timestamp = dataset["timestamp-edge-column"]

    device = torch.device(args.device)

    # Assign user and movie IDs and use them as features (to learn an individual trainable
    # embedding for each entity)
    g.nodes[user_ntype].data["id"] = torch.arange(g.num_nodes(user_ntype))
    g.nodes[item_ntype].data["id"] = torch.arange(g.num_nodes(item_ntype))

    # Prepare torchtext dataset and Vocabulary
    textset = {}
    tokenizer = get_tokenizer(None)

    textlist = []
    batch_first = True

    # 处理每个item节点的title
    item_node_num = g.num_nodes(item_ntype)
    for i in range(item_node_num):
        for key in item_texts.keys():
            l = tokenizer(item_texts[key][i].lower())
            textlist.append(l)

    # 建Vocab——vocab2、全乎的数据textset
    for key, _ in item_texts.items():
        vocab2 = build_vocab_from_iterator(
            textlist, specials=["<unk>", "<pad>"]
        )
        textset[key] = (
            textlist,
            vocab2,
            vocab2.get_stoi()["<pad>"],
            batch_first,
        )

    # Sampler
    # item2item sampler
    # 这名字起的不好，实际上是一个IterableDataset（Dataset的子类）
    batch_sampler_dataset = sampler_module.ItemToItemBatchSampler(
        g, user_ntype, item_ntype, args.batch_size
    )

    # neighbor sampler
    neighbor_sampler = sampler_module.NeighborSampler(
        g,
        user_ntype,
        item_ntype,
        args.random_walk_length,
        args.random_walk_restart_prob,
        args.num_random_walks,
        args.num_neighbors,
        args.num_layers,
    )
    collator = sampler_module.PinSAGECollator(
        neighbor_sampler, g, item_ntype, textset
    )

    # 训练集
    dataloader = DataLoader(
        batch_sampler_dataset,
        collate_fn=collator.collate_train,
        num_workers=args.num_workers,
    )

    # 测试集
    dataloader_test = DataLoader(
        torch.arange(g.num_nodes(item_ntype)),
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers,
    )
    dataloader_it = iter(dataloader)

    # Model
    model = PinSAGEModel(
        g, item_ntype, textset, args.hidden_dims, args.num_layers
    ).to(device)

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print('begin to train.')
    # For each batch of head-tail-negative triplets...
    for epoch_id in range(args.num_epochs):
        model.train()
        for _ in tqdm.trange(args.batches_per_epoch):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            # Copy to GPU
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            # 这里是重点！！！
            loss = model(pos_graph, neg_graph, blocks).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            # item_batches = torch.arange(g.num_nodes(item_ntype)).split(
            #     args.batch_size
            # )
            h_item_batches = []
            for blocks in dataloader_test:
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)

                h_item_batches.append(model.get_repr(blocks))
            h_item = torch.cat(h_item_batches, 0)

            print(evaluation.evaluate_nn(dataset, h_item, args.k, args.batch_size))


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--random-walk-length", type=int, default=2)
    parser.add_argument("--random-walk-restart-prob", type=float, default=0.5)
    parser.add_argument("--num-random-walks", type=int, default=10)
    parser.add_argument("--num-neighbors", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-dims", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device", type=str, default="cpu"
    )  # can also be "cuda:0"
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batches-per-epoch", type=int, default=20000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    # Load dataset.
    # data.pkl: meta info and data;
    # train_g.bin: graph network.
    data_info_path = os.path.join(args.dataset_path, "data.pkl")
    with open(data_info_path, "rb") as f:
        dataset = pickle.load(f)
    train_g_path = os.path.join(args.dataset_path, "train_g.bin")

    # [Graph(num_nodes={'movie': 3706, 'user': 6040},
    #       num_edges={('movie', 'watched-by', 'user'): 988129, ('user', 'watched', 'movie'): 988129},
    #       metagraph=[('movie', 'user', 'watched-by'), ('user', 'movie', 'watched')])]
    g_list, _ = dgl.load_graphs(train_g_path)
    dataset["train-graph"] = g_list[0]

    train(dataset, args)
