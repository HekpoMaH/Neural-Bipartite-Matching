#!/usr/bin/env python
import time
import torch
import argparse
import numpy as np
import random
import sys

def append_flow_edge(edge_index, inv_edge_index, flow, first, second, cap):
    edge_index[0].append(first)
    edge_index[1].append(second)
    flow.append(cap)
    inv_edge_index[first][second]=len(edge_index[0])-1

def init_maxflow_graph(n, s):
    edge_index = [[], []]
    start = 0
    end = n+s+1
    inv_edge_index = torch.full((n+s+2, n+s+2), -100, dtype=torch.int32)
    flow = []

    for i in range(1,n+1):
        append_flow_edge(edge_index, inv_edge_index, flow, start, i, 1)
        append_flow_edge(edge_index, inv_edge_index, flow, i, start, 0)

    for i in range(n+1, n+s+1):
        append_flow_edge(edge_index, inv_edge_index, flow, i, end, 1)
        append_flow_edge(edge_index, inv_edge_index, flow, end, i, 0)

    return edge_index, inv_edge_index, start, end, flow


def read_capacity_mccme(n, s):
    edge_index, start, end, flow = init_maxflow_graph(n, s)

    for i in range(1,s+1):
        for j, x in enumerate([int(x) for x in input().split()]):
            if x == 1:
                append_flow_edge(edge_index, inv_edge_index, flow, j+1, n+i, 1)
                append_flow_edge(edge_index, inv_edge_index, flow, n+i, j+1, 0)
    return edge_index, flow

def read_capacity(n, s, init_from_processed=False):
    if not init_from_processed:
        edge_index, inv_edge_index, start, end, flow = init_maxflow_graph(n, s)
    else:
        edge_index, inv_edge_index, start, end, flow = \
                ([[], []],
                 torch.full((n+s+2, n+s+2), -100, dtype=torch.int32),
                 0,
                 n+s+1,
                 [])
    edge_index_inp = [[], []]
    edge_index_inp[0] = [int(x) for x in input().split()]
    edge_index_inp[1] = [int(x) for x in input().split()]
    if init_from_processed:
        _ = [x for x in input().split()]
        flows = [int(x) for x in input().split()]
    for i in range(len(edge_index_inp[0])):
        flow_to_add = flows[i] if init_from_processed else 1
        append_flow_edge(edge_index, inv_edge_index, flow, edge_index_inp[0][i], edge_index_inp[1][i], flow_to_add)
        if not init_from_processed:
            append_flow_edge(edge_index, inv_edge_index, flow, edge_index_inp[1][i], edge_index_inp[0][i], 0)
    return edge_index, inv_edge_index, flow

def step_bellman_ford(size, bf, pred, edge_index, flow, weights, cached):
    assert pred[0] == 0
    if cached:
        return bf, pred, cached
    inf = 255

    bf_new = bf.clone()
    pred_new = pred.clone()

    start = time.time()
    n1 = edge_index[0]
    n2 = edge_index[1]
    flow = torch.tensor(flow)
    flow_penalty = (1 - flow)*inf
    flow_penalty = torch.max(flow_penalty, torch.zeros_like(flow_penalty))

    potential_new = torch.min(bf[n1] + weights + flow_penalty, torch.full_like(flow_penalty, inf))
    assert ((bf[n1]+weights)[(flow_penalty == 0) & (bf[n1] != 255)] < 255).all(), (bf[n1], weights)
    
    bf_new2 = torch.min(bf_new[n2], potential_new)

    pred_mask = torch.argmin(torch.stack((bf_new[n2], potential_new)), dim=0)
    pred_new2 = torch.where(pred_mask.bool(), torch.tensor(n1, dtype=torch.long), pred_new[n2])

    for i in range(len(bf)):
        mask = torch.tensor(n2) == i
        val, idx = torch.min(bf_new2[mask], 0)
        bf_new[i] = val
        pred_new[i] = pred_new2[mask][idx] if val != inf else -1
        
    if (bf_new == bf).all():
        cached = True

    return bf_new, pred_new, cached

def get_inf(size, edge_index, flow, weights):
    return 255

def find_augmenting_path(size, edge_index, flow, generate_dataset=False):

    weights = torch.randint(1, 16, (len(edge_index[0]),))
    inf = get_inf(size, edge_index, flow, weights)
    if generate_dataset:
        np.savetxt(sys.stdout, weights.view(1, -1), fmt='%s')
        print(*flow, sep=' ')

    pred = torch.full((size,), -1, dtype=torch.long)
    pred[0] = 0

    bf = torch.full((size,), inf, dtype=torch.long)
    bf[0] = 0

    if generate_dataset:
        np.savetxt(sys.stdout, bf.view(1,-1), fmt='%s')
        np.savetxt(sys.stdout, pred.view(1,-1), fmt='%s')
    cached = False
    for i in range(size-1):
        bf, pred, cached = step_bellman_ford(size, bf, pred, edge_index, flow, weights, cached)
        assert pred[0] == 0
        if generate_dataset:
            np.savetxt(sys.stdout, bf.view(1,-1), fmt='%s')
            np.savetxt(sys.stdout, pred.view(1,-1), fmt='%s')

    curr = n+s+1
    if bf[curr] >= inf:
        return [], -1

    path = [curr]
    bottleneck = inf
    while curr != 0:
        predecessor = int(pred[curr])
        path.append(predecessor)
        bottleneck = min(bottleneck, (flow[inv_edge_index[predecessor][curr]]))
        curr = int(pred[curr])

    return list(reversed(path)), bottleneck

def augment_flow(inv_edge_index, flow, path, bottleneck, generate_dataset=False):
    for i in range(len(path)-1):
        flow[inv_edge_index[path[i]][path[i+1]]] -= bottleneck
        flow[inv_edge_index[path[i+1]][path[i]]] += bottleneck

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='First iteration')
    parser.add_argument('--first_iter', action='store_true')
    parser.add_argument('--eval_processed_input', action='store_true')
    args = parser.parse_args()
    n, s = [int(x) for x in input().split()]
    size = n+s+2
    edge_index, inv_edge_index, flow = read_capacity(n, s, args.eval_processed_input)
    if not args.eval_processed_input:
        print(n, s)
        print(*edge_index[0], sep=' ')
        print(*edge_index[1], sep=' ')
    
    path, bottleneck = find_augmenting_path(size, edge_index, flow, generate_dataset=not args.eval_processed_input)

    while path != []:
        augment_flow(inv_edge_index, flow, path, bottleneck)
        if args.first_iter:
            exit(0)
        path, bottleneck = find_augmenting_path(size, edge_index, flow, generate_dataset=not args.eval_processed_input)


    maxflow = 0
    for i in range(size):
        if inv_edge_index[n+s+1][i] != -100:
            maxflow += flow[inv_edge_index[n+s+1][i]]

    if args.eval_processed_input:
        print(maxflow)
