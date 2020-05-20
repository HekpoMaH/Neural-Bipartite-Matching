"""
Usage:
    test.py [options] [--algorithms=ALGO]... MODEL_TO_LOAD

Options:
    -h --help                        Show this screen.
    --algorithms ALGO                Which algorithms to add. One of {AugmentingPath, BFS}
    --processor-type PROC            Type of processor. One of {MPNN, PNA, GAT}. [default: MPNN]
    --upscale UP                     Test on larger data. Remember to add underscore (e.g. _2x) [default: ]
"""

import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch_geometric
from torch_geometric.data import DataLoader

from tqdm import tqdm
from docopt import docopt

from flow_datasets import SingleIterationDataset
from hyperparameters import get_hyperparameters
from utils import get_print_info, iterate_over, load_algorithms, get_print_format
from flow_datasets import SingleIterationDataset
from models import AlgorithmProcessor, AugmentingPathNetwork

args = docopt(__doc__)
args["--use-ints"] = True # Always uses integers
print(args["--algorithms"])
print(args["MODEL_TO_LOAD"])

hyperparameters = get_hyperparameters()
DEVICE = hyperparameters["device"]
DIM_LATENT = hyperparameters["dim_latent"]

processor = AlgorithmProcessor(DIM_LATENT, SingleIterationDataset, args["--processor-type"]).to(DEVICE)
load_algorithms(args["--algorithms"], processor, args["--use-ints"])
processor.load_state_dict(torch.load(args["MODEL_TO_LOAD"]))
processor.eval()

upscale = args["--upscale"]

for algorithm in processor.algorithms.values():
    algorithm.test_dataset = algorithm.dataset_class(algorithm.dataset_root, split='test'+upscale, less_wired=True, device='cpu')
    # print(algorithm.test_dataset[0].x[:, 0, 0])

iterate_over(processor, test=True)
if "AugmentingPath" not in processor.algorithms:
    print("Mean/Last step acc", processor.algorithms["BFS"].get_validation_accuracies())
    exit(0)

fmt = get_print_format()

(total_loss_dist, total_loss_pred, total_loss_term, find_min, total_loss,
 mean_step_acc, final_step_acc, tnr, subtract_acc, broken_invariants,
 broken_reachabilities, broken_flows, broken_all, len_broken) =\
         get_print_info(processor.algorithms["AugmentingPath"])

if get_hyperparameters()["calculate_termination_statistics"]:
    augmenting_path_network = processor.algorithms["AugmentingPath"]
    print("TP", augmenting_path_network.true_positive)
    print("TN", augmenting_path_network.true_negative)
    print("FP", augmenting_path_network.false_positive)
    print("FN", augmenting_path_network.false_negative)
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
    'N/A'))
