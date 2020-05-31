import time
import signal
import torch
import torch_geometric
from torch_geometric.data import DataLoader

import models
import flow_datasets
from hyperparameters import get_hyperparameters
from half_deterministic import obtain_paths
from torch_cluster import random_walk

def get_mask_to_process(continue_p, batch_ids, edge_ids, debug=False):
    """

    Used for graphs with different number of steps needed to be performed

    Returns:
    mask (1d tensor): The mask for which nodes still need to be processed

    """
    if debug:
        print("Getting mask processing")
        print("Continue p:", continue_p)
    mask = continue_p[batch_ids] > 0.5
    edge_mask = mask[edge_ids] > 0.5 
    if debug:
        print("Mask:", mask)
    return mask, edge_mask

def get_adj_flow_matrix(size, edge_index, capacities):
    return (torch_geometric.utils.to_dense_adj(edge_index).squeeze().bool(),
            torch_geometric.utils.to_dense_adj(edge_index, edge_attr=capacities).squeeze())

def flip_edge_index(edge_index):
    return torch.stack((edge_index[1], edge_index[0]), dim=0)

def get_true_termination(batch, x_curr, y_curr):
    """ Gets termination values per each graph"""
    true_termination = torch.stack(
        [(~(x_curr[batch.batch == btch] == y_curr[batch.batch == btch]).all()).float()
            for btch in range(batch.num_graphs)],
    )
    return true_termination

def split_per_graph(batch_ids, to_split, num_graphs=None):
    """ Splits a value into subvalues per each graph """
    if num_graphs is None:
        num_graphs = batch_ids.max()+1
    splitted = torch.stack([to_split[batch_ids == btch] for btch in range(num_graphs)])
    return splitted

def get_graph_embedding(batch_ids, latent_nodes, GRAPH_SIZES, reduction='mean'):
    """ Gets the embedding of each graph in batch """
    graph_embs = split_per_graph(batch_ids, latent_nodes, num_graphs=len(GRAPH_SIZES)).sum(1)
    graph_embs /= GRAPH_SIZES.unsqueeze(1)
    return graph_embs

def interrupted(_interrupted=[False], _default=[None]):
    if _default[0] is None or signal.getsignal(signal.SIGINT) == _default[0]:
        _interrupted[0] = False
        def handle(signal, frame):
            if _interrupted[0] and _default[0] is not None:
                _default[0](signal, frame)
            print('Interrupt!')
            _interrupted[0] = True
        _default[0] = signal.signal(signal.SIGINT, handle)
    return _interrupted[0]

def add_self_loops(batch):
    edge_attr = batch.edge_attr[:, 0]
    edge_cap = batch.edge_attr[:, 1]
    new_edge_index, edge_attr = torch_geometric.utils.add_self_loops(batch.edge_index, edge_attr, fill_value=0)
    _, edge_cap = torch_geometric.utils.add_self_loops(batch.edge_index, edge_cap, fill_value=0)
    batch.edge_attr = torch.cat((edge_attr.view(-1, 1), edge_cap.view(-1, 1)), dim=1)
    batch.edge_index = new_edge_index
    return batch

def get_sizes_and_source_sink(batch):
    DEVICE = get_hyperparameters()["device"]
    GRAPH_SIZES = torch.unique(batch.batch, return_counts=True)[1].to(DEVICE)
    SOURCE_NODES = (GRAPH_SIZES.cumsum(0)-GRAPH_SIZES).clone().detach()
    SINK_NODES = (GRAPH_SIZES.cumsum(0)-1).clone().detach()
    return GRAPH_SIZES, SOURCE_NODES, SINK_NODES

def finish(x, y, batch_ids, steps, STEPS_SIZE, GRAPH_SIZES):
    """
    Returns whether it's a final iteration or not in real task

    Returns true/false value per graph (as a mask)

    N.B. Not what the network thinks
    """
    DEVICE = get_hyperparameters()["device"]
    if steps == 0:
        return torch.ones(len(GRAPH_SIZES), device=DEVICE)
    if not steps < STEPS_SIZE-1:
        return torch.zeros(len(GRAPH_SIZES), device=DEVICE)
    x_curr = torch.index_select(x, 1, torch.tensor([steps], dtype=torch.long, device=DEVICE)).squeeze(1).to(DEVICE)
    y_curr = torch.index_select(y, 1, torch.tensor([steps], dtype=torch.long, device=DEVICE)).squeeze(1).to(DEVICE)
    noteq = (~(x_curr == y_curr))
    hyperparameters = get_hyperparameters()
    batches_inside = batch_ids.max()+1
    noteq_batched = noteq.view(batches_inside, -1, hyperparameters["dim_target"])
    true_termination = noteq_batched.any(dim=1).any(dim=-1).float()
    return true_termination

def get_input(batch, EPSILON, train, x_curr, last_output): # Always returns last output
    inp = last_output

    assert not x_curr.requires_grad
    assert not x_curr[:, 0].requires_grad
    return inp

def get_print_info(augmenting_path_network):
    total_loss_dist, total_loss_pred, total_loss_term, findmin = augmenting_path_network.get_validation_losses()
    mean_step, final_step, tnr, subtract_acc = augmenting_path_network.get_validation_accuracies()
    total_loss = total_loss_dist + total_loss_pred + total_loss_term
    broken_invariants, broken_reachabilities, broken_flows, broken_all = augmenting_path_network.get_broken_invariants()
    len_broken = len(broken_invariants)
    return total_loss_dist, total_loss_pred, total_loss_term, findmin, total_loss, mean_step, final_step, tnr, subtract_acc, broken_invariants, broken_reachabilities, broken_flows, broken_all, len_broken


def iterate_over(processor, optimizer=None, test=False):
    hyperparameters = get_hyperparameters()
    DEVICE = hyperparameters["device"]
    BATCH_SIZE = hyperparameters["batch_size"]

    for algorithm in processor.algorithms.values():
        if processor.training:

            algorithm.iterator = iter(DataLoader(algorithm.train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=8))
        else:
            algorithm.iterator = iter(DataLoader(algorithm.test_dataset if test
                                                 else algorithm.val_dataset, batch_size=BATCH_SIZE,
                                                 shuffle=False, drop_last=False, num_workers=8))

    if not processor.training:
        for algorithm in processor.algorithms.values():
            algorithm.zero_validation_stats()
    try:
        while True:
            for algorithm in processor.algorithms.values():
                batch = next(algorithm.iterator)
                batch.to(DEVICE)
                EPS_I = 0
                start = time.time()
                with torch.set_grad_enabled(processor.training):
                    output = algorithm.process(batch, EPS_I)
                    if not processor.training:
                        algorithm.update_validation_stats(batch, output)

            if processor.training:
                processor.update_weights(optimizer)
            if interrupted():
                break
    except StopIteration: # datasets should be the same size
        pass

    for algorithm in processor.algorithms.values(): # for when they are not
        if not processor.training:
            algorithm.zero_tracking_losses_and_statistics()
        try:
            while True:
                batch = next(algorithm.iterator)
                batch.to(DEVICE)
                EPS_I = 0
                start = time.time()
                with torch.set_grad_enabled(processor.training):
                    output = algorithm.process(batch, EPS_I)
                    if not processor.training:
                        algorithm.update_validation_stats(batch, output)
                if processor.training:
                    processor.update_weights(optimizer)
        except StopIteration:
            pass


def load_algorithms(algorithms, processor, use_ints):
    hyperparameters = get_hyperparameters()
    DEVICE = hyperparameters["device"]
    DIM_LATENT = hyperparameters["dim_latent"]
    DIM_NODES_BFS = hyperparameters["dim_nodes_BFS"]
    DIM_NODES_AugmentingPath = hyperparameters["dim_nodes_AugmentingPath"]
    DIM_EDGES = hyperparameters["dim_edges"]
    DIM_EDGES_BFS = hyperparameters["dim_edges_BFS"]
    DIM_BITS = hyperparameters["dim_bits"] if use_ints else None
    for algorithm in algorithms:
        if algorithm == "AugmentingPath":
            algo_net = models.AugmentingPathNetwork(DIM_LATENT, DIM_NODES_AugmentingPath, DIM_EDGES, processor, flow_datasets.SingleIterationDataset, './all_iter', bias=hyperparameters["bias"], use_ints=use_ints, bits_size=DIM_BITS).to(DEVICE)
        if algorithm == "BFS":
            algo_net = models.BFSNetwork(DIM_LATENT, DIM_NODES_BFS, DIM_EDGES_BFS, processor, flow_datasets.BFSSingleIterationDataset, './bfs').to(DEVICE)
        processor.add_algorithm(algo_net, algorithm)

def integer2bit(integer, num_bits=8):
    # Credit: https://github.com/KarenUllrich/pytorch-binary-converter/blob/master/binary_converter.py
    """Turn integer tensor to binary representation.
        Args:
            integer : torch.Tensor, tensor with integers
            num_bits : Number of bits to specify the precision. Default: 8.
        Returns:
            Tensor: Binary tensor. Adds last dimension to original tensor for
            bits.
    """
    dtype = integer.type()
    exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
    exponent_bits = exponent_bits.repeat(integer.shape + (1,))
    out = integer.unsqueeze(-1) / 2 ** exponent_bits
    return (out - (out % 1)) % 2

_POWERS_OF_2 = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], device=get_hyperparameters()["device"])
def bit2integer(bit_logits):
    bits = (bit_logits > 0).long()
    bits *= _POWERS_OF_2
    ints = bits.sum(dim=1).float()
    return ints

def create_inv_edge_index(batch_size, size, edge_index):
    iei = torch.full((batch_size*size, batch_size*size), -100, dtype=torch.long)
    for i in range(len(edge_index[0])):
        iei[edge_index[0][i]][edge_index[1][i]] = i
    return iei

def get_walks_from_output(output, GRAPH_SIZES, SOURCE_NODES, SINK_NODES):
    path_matrix, stop_move_backward_col, final = obtain_paths(
        output,
        GRAPH_SIZES,
        GRAPH_SIZES.max(),
        SOURCE_NODES,
        SINK_NODES,
        return_path_matrix=True
    )
    mask = (path_matrix == -100)
    path_matrix = path_matrix.where(~mask, final.repeat(GRAPH_SIZES.max(), 1).t())
    return path_matrix, stop_move_backward_col, mask

def get_walks(training, batch, output, GRAPH_SIZES, SOURCE_NODES, SINK_NODES):
    if training:
        return random_walk(batch.edge_index[0], batch.edge_index[1], SINK_NODES, walk_length=get_hyperparameters()["walk_length"], coalesced=True).long(), None
    path_matrix, _, mask = get_walks_from_output(
        output,
        GRAPH_SIZES,
        SOURCE_NODES,
        SINK_NODES
    )
    return path_matrix, mask

def get_print_format():
    fmt = """
==========================
Mean step acc: {:.4f}    Last step acc: {:.4f}
Mincap TNR: {:.4f}
Subtract accuracy: {:.4f}
loss-(dist,pred,term,findmin,total): {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}
    broken_invariants: {:2d}/{:3d}
    broken_all: {:2d}/{:3d}
    broken_reachabilities: {:2d}/{:3d}
    broken_flows: {:2d}/{:3d}
patience: {}
===============
"""
    return fmt
