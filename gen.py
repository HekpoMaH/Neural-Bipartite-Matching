#!/usr/bin/env python
import random
import torch
import argparse

parser = argparse.ArgumentParser(description='Node limits')
parser.add_argument('s_limit', type=int)
parser.add_argument('n_limit', type=int)
parser.add_argument('probp', type=int)
parser.add_argument('probq', type=int)
parser.add_argument('--random_node_amount', action='store_true')
args = parser.parse_args()
n_limit, s_limit = (args.n_limit, args.s_limit)
n = random.randrange(n_limit-1) + 2 if args.random_node_amount else n_limit
s = random.randrange(s_limit-1) + 2 if args.random_node_amount else s_limit

wired_factor = float(args.probp)/float(args.probq)
matrix = torch.rand(s, n) < wired_factor

edge_index = [[], []]
for i, row in enumerate(matrix):
    for j, cell in enumerate(row):
        if cell:
            edge_index[0].append(i+1)
            edge_index[1].append(j+1 + len(matrix))

print(s, n)
print(*edge_index[0], sep=' ')
print(*edge_index[1], sep=' ')
