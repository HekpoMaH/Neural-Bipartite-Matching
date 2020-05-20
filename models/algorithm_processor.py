import time
import torch.nn as nn

from pprint import pprint

from layers import MPNN, PNAWrapper, GAT
from hyperparameters import get_hyperparameters
from flow_datasets import SingleIterationDataset

class AlgorithmProcessor(nn.Module):
    def __init__(self, latent_features, dataset, processor_type='MPNN'):
        assert processor_type in ['MPNN', 'PNA', 'GAT']
        super(AlgorithmProcessor, self).__init__()
        if processor_type == 'MPNN':
            self.processor = MPNN(latent_features, latent_features, latent_features, bias=get_hyperparameters()["bias"])
        elif processor_type == 'PNA':
            self.processor = PNAWrapper(latent_features, latent_features, latent_features, SingleIterationDataset, bias=get_hyperparameters()["bias"])
        elif processor_type == 'GAT':
            self.processor = GAT(latent_features, latent_features, latent_features, bias=get_hyperparameters()["bias"])
        self.algorithms = nn.ModuleDict()

    def add_algorithm(self, algorithm, name):
        self.algorithms[name] = algorithm

    def update_weights(self, optimizer):
        loss = 0
        for name, algorithm in self.algorithms.items():
            print("Algorithm", name)
            losses_dict =\
                    algorithm.get_losses_dict()
            pprint(losses_dict)
            loss += algorithm.get_training_loss()
            if get_hyperparameters()["calculate_termination_statistics"]: #DEPRECATED
                print("Term precision:", algorithm.true_positive/(algorithm.true_positive+algorithm.false_positive) if algorithm.true_positive+algorithm.false_positive else 'N/A')
                print("Term recall:", algorithm.true_positive/(algorithm.true_positive+algorithm.false_negative) if algorithm.true_positive+algorithm.false_negative else 'N/A')

        start = time.time()
        optimizer.zero_grad()
        print("LOSSITEM", loss.item())
        loss.backward()
        optimizer.step()

    def reset_all_weights(self): # Debugging tool
        for name, W in self.named_parameters():
            W.data.fill_(0.01)

    def get_sum_grad(self): # Debugging tool
        s = 0
        for name, W in self.named_parameters(): 
            print(name, W.grad)
            s += W.grad.sum()
        return s

