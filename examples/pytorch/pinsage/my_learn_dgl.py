# coding: utf-8
import dgl
import torch
g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
g.edata['h'] = torch.ones(2, 1)
print(g.edata['h'])
print(g.srcdata)
print(g.ndata)