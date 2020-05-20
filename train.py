"""
Usage:
    train.py [options] [--algorithms=ALGO]...

Options:
    -h --help                        Show this screen.
    --algorithms ALGO                Which algorithms to add. One of {AugmentingPath, BFS}
    --model-name NAME                Specific name of model
    --processor-type PROC            Type of processor. One of {MPNN, PNA, GAT}. [default: MPNN]
"""
from datetime import datetime

import time
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from torch_geometric.data import DataLoader

import numpy as np
from pprint import pprint
from docopt import docopt


import models
from flow_datasets import SingleIterationDataset
from utils import get_print_info, get_print_format, interrupted, iterate_over, load_algorithms
from hyperparameters import get_hyperparameters


if __name__ == "__main__":

    args = docopt(__doc__)
    hyperparameters = get_hyperparameters()
    DEVICE = hyperparameters["device"]
    DIM_LATENT = hyperparameters["dim_latent"]
    DIM_EDGES = hyperparameters["dim_edges"]
    NAME = args["--model-name"] if args["--model-name"] is not None else datetime.now().strftime("%b-%d-%Y-%H-%M")

    processor = models.AlgorithmProcessor(DIM_LATENT, SingleIterationDataset, args["--processor-type"]).to(DEVICE)
    print("PARAMETERS", sum(p.numel() for p in processor.parameters()))
    print(list((name, p.numel()) for name, p in processor.named_parameters()))
    load_algorithms(args["--algorithms"], processor, True)
    # processor.reset_all_weights()
    params = list(processor.parameters())
    print(DEVICE)
    print(processor)
    augmenting_path_network = None
    for key, algorithm in processor.algorithms.items():
        if type(algorithm) == models.AugmentingPathNetwork:
            augmenting_path_network = algorithm
    print(augmenting_path_network)

    BATCH_SIZE = hyperparameters["batch_size"]
    PATIENCE_LIMIT = hyperparameters["patience_limit"]
    GROWTH_RATE = hyperparameters["growth_rate_sigmoid"]
    SIGMOID_OFFSET = hyperparameters["sigmoid_offset"]

    patience = 0
    last_mean = 0
    last_final = 0
    last_broken = 100
    last_loss = 0*1e9 if augmenting_path_network is not None else 1e9
    cnt = 0

    fmt = get_print_format()

    best_model = models.AlgorithmProcessor(DIM_LATENT, SingleIterationDataset, args["--processor-type"]).to(DEVICE)
    best_model.algorithms = nn.ModuleDict(processor.algorithms.items())
    best_model.load_state_dict(copy.deepcopy(processor.state_dict()))

    torch.set_printoptions(precision=20)

    with torch.autograd.profiler.profile(enabled=False, use_cuda=True) as prof:
        # for algorithm in processor.algorithms:
        #     algorithm.loader = DataLoader(algorithm.train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=8)
        #     algorithm.val_loader = DataLoader(algorithm.val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=8)
        optimizer = optim.Adam(params, lr=hyperparameters["lr"], weight_decay=hyperparameters["weight_decay"])
        for epoch in range(3000):# FIXME
            if interrupted():
                break
            # 0.0032
            processor.train()
            iterate_over(processor, optimizer)

            patience += 1
            print('Epoch {:4d}: \n'.format(epoch), end=' ')
            processor.eval()
            iterate_over(processor)
            # print("Mean/Last step acc", processor.algorithms[0].get_validation_accuracies())
            # total_loss = sum(processor.algorithms[0].get_validation_losses()) #TODO PRETTIER!

            if augmenting_path_network is None:
                total_loss = sum(processor.algorithms["BFS"].get_validation_losses())
                if (total_loss) < last_loss:
                    patience = 0
                    last_loss = (total_loss)
                    best_model.load_state_dict(copy.deepcopy(processor.state_dict()))
                print("Total Loss:", total_loss, "Patience:", patience)
            
            if augmenting_path_network is not None: # TODO prettier code plz
                (total_loss_dist, total_loss_pred, total_loss_term, find_min, total_loss,
                 mean_step_acc, final_step_acc, tnr, subtract_acc, broken_invariants,
                 broken_reachabilities, broken_flows, broken_all, len_broken) =\
                         get_print_info(processor.algorithms["AugmentingPath"])

                if get_hyperparameters()["calculate_termination_statistics"]: #DEPRECATED
                    print("Term precision:",
                            augmenting_path_network.true_positive/(augmenting_path_network.true_positive+augmenting_path_network.false_positive)
                            if
                            augmenting_path_network.true_positive+augmenting_path_network.false_positive
                            else 'N/A')
                    print("Term recall:",
                            augmenting_path_network.true_positive/(augmenting_path_network.true_positive+augmenting_path_network.false_negative)
                            if
                            augmenting_path_network.true_positive+augmenting_path_network.false_negative
                            else 'N/A')

                if (final_step_acc) > last_loss:
                    patience = 0
                    last_loss = (final_step_acc)
                    best_model.load_state_dict(copy.deepcopy(processor.state_dict()))

                print(fmt.format(
                    mean_step_acc,
                    final_step_acc,
                    tnr,
                    subtract_acc,
                    total_loss_dist,
                    total_loss_pred,
                    total_loss_term,
                    find_min,
                    total_loss,
                    sum(broken_invariants),
                    len_broken,
                    sum(broken_all),
                    len_broken,
                    sum(broken_reachabilities),
                    len_broken,
                    sum(broken_flows),
                    len_broken,
                    patience))

            torch.save(processor.state_dict(), './.serialized_models/test_'+NAME+'_epoch_'+str(epoch)+'.pt')

            if patience >= PATIENCE_LIMIT:
                break

    torch.save(best_model.state_dict(), './.serialized_models/best_'+NAME+'.pt')
