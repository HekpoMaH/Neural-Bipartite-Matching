#!/usr/bin/env python
"""
Usage:
    half_deterministic.py [options] [--algorithms=ALGO]... MODEL_TO_LOAD SAVE_FILE

Options:
    -h --help                        Show this screen.
    --algorithms ALGO                Which algorithms to add. One of {AugmentingPath, BFS}
    --processor-type PROC            Type of processor. One of {MPNN, PNA, GAT}. [default: MPNN]
    --use-BFS-for-termination        Use BFS for deciding if more augmenting paths exist. Remember to load BFS algorithms [default: False]
    --use-neural-bottleneck          Use the network to extract the bottleneck [default: False]
    --use-neural-augmentation        Use the network to provide the new forward capacities after augmenting the flow. (Backward capacity is total edge capacity minus forward) [default: False]
    --threshold X                    How many times to break an invariant before it is considered that no more augmenting paths exist [default: 1]
    --upscale UP                     Test on larger data. Remember to add underscore (e.g. _2x) [default: ]
    --probp P                        Probability P (P/Q) wired factor [default: 1]
    --probq Q                        Probability Q (P/Q) wired factor [default: 4]
    --use-ints                       Does the dataset use integers [default: True]
"""

import torch
import time
import argparse
import numpy as np
import random
import sys
import copy
from docopt import docopt
from tqdm import tqdm

from torch_geometric.data import DataLoader

import models
import utils

from deterministic import append_flow_edge, read_capacity, get_inf
from flow_datasets import GraphOnlyDataset, GraphOnlyDatasetBFS, SingleIterationDataset
from hyperparameters import get_hyperparameters

def obtain_paths(predecessors, GRAPH_SIZES, STEPS_SIZE, SOURCE_NODES, SINK_NODES, return_path_matrix=False):
    hyperparameters = get_hyperparameters()
    DEVICE = hyperparameters["device"]
    path_matrix = torch.full((len(GRAPH_SIZES), STEPS_SIZE), -100, device=DEVICE, dtype=torch.long)
    stop_move_backward_col = torch.zeros(len(GRAPH_SIZES), device=DEVICE, dtype=torch.long)
    final = SOURCE_NODES.clone()

    for i, n in enumerate(SINK_NODES):
        path_matrix[i][0] = n.item()

    for i in range(1, STEPS_SIZE):
        rowcols = (range(len(GRAPH_SIZES)), stop_move_backward_col)
        upd = (path_matrix[rowcols] != SOURCE_NODES)
        upd2 = path_matrix[rowcols] != predecessors[path_matrix[rowcols]]
        upd3 = predecessors[path_matrix[rowcols]] != -1
        upd &= upd2 & upd3
        path_matrix[upd, i] = predecessors[path_matrix[upd, i-1]]
        stop_move_backward_col[upd] = i
        final[upd] = path_matrix[upd, i]
    path = torch.stack([torch.stack([path_matrix[i][:-1], path_matrix[i][1:]], dim=0) for i in range(len(GRAPH_SIZES))], dim=0)
    return path_matrix if return_path_matrix else path, stop_move_backward_col, final

def get_pairs(i, stop_move_backward_col, path, do_not_process):
    mask = (i < stop_move_backward_col) & (~do_not_process)
    index = torch.tensor(i, device=get_hyperparameters()["device"], dtype=torch.long)
    pairs = (path[mask][:]).index_select(dim=2, index=index).squeeze(-1)
    return pairs

def zero_unreachable_and_frozen_bottleneck(bottleneck, do_not_process, reachable_sinks):
    bottleneck[do_not_process] = 0
    bottleneck[~reachable_sinks] = 0

def get_bottleneck(path, stop_move_backward_col, flow, inv_edge_index, do_not_process, reachable_sinks):
    largest_len = max(stop_move_backward_col)
    bottleneck = torch.full_like(stop_move_backward_col, 100)
    for i in range(largest_len):
        mask = (i < stop_move_backward_col) & (~do_not_process)
        pairs = get_pairs(i, stop_move_backward_col, path, do_not_process)
        edge_idx = inv_edge_index[(pairs[:, 1], pairs[:, 0])]
        edge_idx_rev = inv_edge_index[(pairs[:, 0], pairs[:, 1])]
        assert (edge_idx != -100).all()
        assert (edge_idx_rev != -100).all()
        bottleneck[mask] = torch.min(bottleneck[mask], flow[edge_idx].long())
    bottleneck[bottleneck == 100] = 0
    zero_unreachable_and_frozen_bottleneck(bottleneck, do_not_process, reachable_sinks)
    return bottleneck
        
def termination_condition(args, i, threshold, batch, reachable_sinks=None):
    if not args["--use-BFS-for-termination"]:
        return i < threshold
    return i < get_hyperparameters()["max_threshold"] and reachable_sinks.any()

def get_redo_mask(use_bfs, use_neural_bottleneck, final, SOURCE_NODES, bottleneck, do_not_process, reachable):
    redo_mask = ((final != SOURCE_NODES) | (bottleneck == 0)) & (~do_not_process)
    if use_bfs:
        redo_mask = reachable & redo_mask
    return redo_mask

def reweight_batch(batch, batch_bfs, use_ints):
    DEVICE = get_hyperparameters()["device"]
    if use_ints:
        weights = torch.randint_like(batch.edge_attr[:, 1], 1, 16, device=DEVICE, dtype=torch.float)
    else:
        weights = 0.8*torch.rand_like(batch.edge_attr[:, 1], device=DEVICE, dtype=torch.float) + 0.2
    batch.edge_attr = torch.stack((weights, batch.edge_attr[:, 1]), dim=1)
    batch_bfs.edge_attr = torch.stack((weights, batch_bfs.edge_attr[:, 1]), dim=1)

def find_augmenting_path(args, batch, batch_bfs, do_not_process, processor, inv_edge_index, threshold, debug=False):
    DEVICE = get_hyperparameters()["device"]
    GRAPH_SIZES, SOURCE_NODES, SINK_NODES = utils.get_sizes_and_source_sink(batch)
    STEPS_SIZE = GRAPH_SIZES.max()
    redo_mask = torch.ones_like(GRAPH_SIZES, device=DEVICE, dtype=torch.bool) & (~do_not_process)
    predecessors_last = torch.full_like(batch.batch, -1, device=DEVICE)
    reachable_last = torch.ones_like(predecessors_last, dtype=torch.bool)
    weights = batch.edge_attr[:, 0]
    flow = batch.edge_attr[:, 1]
    final = SINK_NODES.clone()
    bottleneck = torch.zeros(batch.num_graphs, device=DEVICE)
    path_matrix = torch.full((batch.num_graphs, STEPS_SIZE), -100, device=DEVICE, dtype=torch.long)
    stop_move_backward_col = torch.zeros(batch.num_graphs, device=DEVICE, dtype=torch.long)
    wrong_bottleneck_mask = None
    for algorithm in processor.algorithms.values():
        algorithm.zero_validation_stats()

    if args["--use-BFS-for-termination"]:
        with torch.no_grad():
            reachable = processor.algorithms["BFS"].process(
                batch_bfs,
                EPSILON=0,
                enforced_mask=redo_mask,
                compute_losses_and_broken=False
            )
    else:
        reachable = torch.ones(batch.num_nodes, device=DEVICE, dtype=torch.bool)
    i = 0
    while termination_condition(args, i, threshold, batch, reachable[SINK_NODES]):
        i += 1

        start = time.time()
        with torch.no_grad():
            predecessors = processor.algorithms["AugmentingPath"].process(
                batch,
                EPSILON=0,
                enforced_mask=redo_mask,
                compute_losses_and_broken=False
            )

        predecessors = torch.where(redo_mask[batch.batch], predecessors, predecessors_last)
        predecessors_last = predecessors
        predecessors = predecessors_last.clone()
        path_matrix, stop_move_backward_col, final = obtain_paths(predecessors, GRAPH_SIZES, STEPS_SIZE, SOURCE_NODES, SINK_NODES)
        if args["--use-neural-bottleneck"]:
            walks, mask_end_of_path = utils.get_walks(False, batch, predecessors, GRAPH_SIZES, SOURCE_NODES, SINK_NODES)
            bottleneck = processor.algorithms["AugmentingPath"].find_mins(
                    batch, walks, mask_end_of_path, GRAPH_SIZES, SOURCE_NODES, SINK_NODES
            )
            zero_unreachable_and_frozen_bottleneck(bottleneck, do_not_process, reachable[SINK_NODES])
            real_bottleneck = get_bottleneck(path_matrix, stop_move_backward_col, flow, inv_edge_index, do_not_process, reachable[SINK_NODES])
            wrong_bottleneck_mask = (bottleneck == 1) & (real_bottleneck == 0)
            do_not_process |= wrong_bottleneck_mask
        else:
            bottleneck = get_bottleneck(path_matrix, stop_move_backward_col, flow, inv_edge_index, do_not_process, reachable[SINK_NODES])

        reweight_batch(batch, batch_bfs, args["--use-ints"])
        if args["--use-BFS-for-termination"]:
            with torch.no_grad():
                reachable = processor.algorithms["BFS"].process(
                    batch_bfs,
                    EPSILON=0,
                    enforced_mask=redo_mask,
                    compute_losses_and_broken=False
                )

        reachable = torch.where(redo_mask[batch.batch], reachable, reachable_last)
        reachable_last = reachable
        reachable = reachable_last.clone()
        reachable_sinks = reachable[SINK_NODES]
        redo_mask = get_redo_mask(
            args["--use-BFS-for-termination"], args["--use-neural-bottleneck"],
            final, SOURCE_NODES, bottleneck, do_not_process, reachable_sinks
        )
        do_not_process[redo_mask] |= (~reachable_sinks[redo_mask])

        if debug:
            if (bottleneck == 0).any():
                print("Broke flow cap invariant", file=sys.stderr)
            if (final != SOURCE_NODES).any():
                print("Broke reachability invariant", file=sys.stderr)

        if not redo_mask.any():
            bottleneck[do_not_process] = 0
            return path_matrix, stop_move_backward_col, bottleneck

    bottleneck[redo_mask] = 0
    bottleneck[do_not_process] = 0 # Hack to set redo mask to false for next iterations
    return path_matrix, stop_move_backward_col, bottleneck

def augment_flow(batch, inv_edge_index, path, stop_move_backward_col, bottleneck, do_not_process, use_neural_augmentation, augmenting_path_network=None):
    def get_edge_indexes(step):
        pairs = get_pairs(step, stop_move_backward_col, path, do_not_process)
        edge_idx = inv_edge_index[(pairs[:, 1], pairs[:, 0])]
        edge_idx_rev = inv_edge_index[(pairs[:, 0], pairs[:, 1])]
        return edge_idx, edge_idx_rev

    flow = batch.edge_attr[:, 1]
    old_flow = flow.clone()
    path_matrix = torch.cat((path[:, 0, :], path[:, 1, -1].unsqueeze(-1)), dim=-1)
    mask_end_of_path = path_matrix == -100
    path_matrix[mask_end_of_path] = 0
    if use_neural_augmentation:
        new_flows = augmenting_path_network.augment_flow(batch, path_matrix, mask_end_of_path, bottleneck)
    largest_len = max(stop_move_backward_col)
    needs_rerun = torch.zeros(batch.num_graphs, device=get_hyperparameters()["device"], dtype=torch.bool)
    for i in range(largest_len):
        mask = (i < stop_move_backward_col) & (~do_not_process)
        if not mask.any():
            break
        edge_idx, edge_idx_rev = get_edge_indexes(i)
        assert (edge_idx != -100).all()
        assert (edge_idx_rev != -100).all()
        assert (abs(flow[edge_idx]) <= 1).all(), flow[edge_idx][~(abs(flow[edge_idx]) <= 1)]
        assert (abs(bottleneck[mask]) <= 1).all()
        assert (bottleneck[do_not_process] == 0).all()
        assert (bottleneck[~do_not_process] == 1).all()
        assert (bottleneck[mask] == 1).all()
        if use_neural_augmentation:
            new_flow = new_flows[mask[stop_move_backward_col > 0]][:, i].float()
            should_be = flow[edge_idx] - bottleneck[mask]
            not_what_should_be = (new_flow != should_be)
            needs_rerun[mask] |= not_what_should_be
            consts = flow[edge_idx]+flow[edge_idx_rev]
            new_flow_rev = consts - new_flow
            flow[edge_idx] = torch.where(not_what_should_be, flow[edge_idx], new_flow)
            flow[edge_idx_rev] = torch.where(not_what_should_be, flow[edge_idx_rev], new_flow_rev)
        else:
            flow[edge_idx] -= bottleneck[mask]
            flow[edge_idx_rev] += bottleneck[mask]
    if needs_rerun.any():
        for i in range(largest_len):
            mask = (i < stop_move_backward_col) & (~do_not_process)
            if not mask.any():
                break
            edge_idx, edge_idx_rev = get_edge_indexes(i)
            flow[edge_idx] = old_flow[edge_idx]
            flow[edge_idx_rev] = old_flow[edge_idx_rev]
    return needs_rerun

def run(args, threshold, processor, probp=1, probq=4, savefile=True):
    hyperparameters = get_hyperparameters()
    DEVICE = hyperparameters["device"]
    DIM_LATENT = hyperparameters["dim_latent"]
    if savefile:
        with open(args["SAVE_FILE"], "w"):
            pass
        f = open(args["SAVE_FILE"], "a+")
    dataset = GraphOnlyDataset('./graph_only', split='test'+args["--upscale"], less_wired=True, probp=probp, probq=probq, device='cpu')
    dataset_BFS = GraphOnlyDatasetBFS('./graph_only_BFS', split='test'+args["--upscale"],probp=probp, probq=probq, less_wired=True, device='cpu')
    result_maxflows = []
    with torch.no_grad():
        for rep in range(10):
            print(rep)
            current_result = []
            loader = DataLoader(dataset, batch_size=hyperparameters["batch_size"], shuffle=False, drop_last=False, num_workers=0)
            loader_BFS = iter(DataLoader(dataset_BFS, batch_size=hyperparameters["batch_size"], shuffle=False, drop_last=False, num_workers=0))
            for batch in tqdm(loader, dynamic_ncols=True):
                batch_bfs = next(loader_BFS)
                batch_bfs.to(DEVICE)
                batch.to(DEVICE)
                start_iter = time.time()
                start = time.time()
                GRAPH_SIZES, SOURCE_NODES, SINK_NODES = utils.get_sizes_and_source_sink(batch)
                # we make at most |V|-1 steps
                STEPS_SIZE = GRAPH_SIZES.max()
                inv_edge_index = utils.create_inv_edge_index(len(GRAPH_SIZES), GRAPH_SIZES.max(), batch.edge_index)
                
                do_not_process = torch.zeros_like(GRAPH_SIZES, dtype=torch.bool, device=DEVICE)
                start = time.time()
                path_matrix, stop_move_backward_col, bottleneck = find_augmenting_path(args, batch, batch_bfs, do_not_process, processor, inv_edge_index, threshold)
                do_not_process = bottleneck == 0

                cnt = 0
                while (bottleneck != 0).any():
                    wrong_minus = augment_flow(batch, inv_edge_index, path_matrix, stop_move_backward_col, bottleneck, do_not_process, args["--use-neural-augmentation"], augmenting_path_network=processor.algorithms["AugmentingPath"])
                    batch_bfs.edge_attr[:, 1] = batch.edge_attr[:, 1]
                    do_not_process |= wrong_minus

                    path_matrix, stop_move_backward_col, bottleneck = find_augmenting_path(args, batch, batch_bfs, do_not_process, processor, inv_edge_index, threshold)
                    do_not_process |= (bottleneck == 0)
                    bottleneck[do_not_process] = 0
                    assert ((bottleneck <= 1) & (bottleneck >= 0)).all(), (bottleneck, path_matrix)

                    cnt += 1

                start = time.time()
                maxflows = [0 for sn in SINK_NODES]
                cnt = 0
                for isn, sn in enumerate(SINK_NODES):
                    cnt = 0
                    for i in range(len(batch.batch)):
                        if inv_edge_index[sn][i] != -100:
                            cnt += 1
                            assert 0 <= batch.edge_attr[inv_edge_index[sn][i]][1] <= 1
                            maxflows[isn] += batch.edge_attr[inv_edge_index[sn][i]][1]

                if savefile:
                    print(*[int(mf.item()) for mf in maxflows], sep=' ', end=' ', file=f)
                else:
                    current_result.extend([int(mf.item()) for mf in maxflows])
            if savefile:
                f.write('\n')
            else:
                result_maxflows.append(current_result)
    if savefile:
        f.close()
    else:
        return result_maxflows

if __name__ == "__main__":
    hyperparameters = get_hyperparameters()
    DEVICE = hyperparameters["device"]
    DIM_LATENT = hyperparameters["dim_latent"]
    args = docopt(__doc__)
    print("ARGS", args["--algorithms"])
    processor = models.AlgorithmProcessor(DIM_LATENT, SingleIterationDataset, args["--processor-type"]).to(DEVICE)
    utils.load_algorithms(args["--algorithms"], processor, args["--use-ints"])
    processor.load_state_dict(torch.load(args["MODEL_TO_LOAD"]))
    processor.eval()

    with torch.no_grad():
        run(args, int(args["--threshold"]), processor, int(args["--probp"]), int(args["--probq"]))
