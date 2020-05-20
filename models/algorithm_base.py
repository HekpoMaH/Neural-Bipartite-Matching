import time
import torch
import torch.nn as nn
import torch_geometric

from layers import DecoderNetwork
import utils
import models
from hyperparameters import get_hyperparameters

class AlgorithmBase(nn.Module):

    @staticmethod
    def get_masks(train, batch, to_process, continue_p, enforced_mask):
        if train:
            mask, edge_mask = utils.get_mask_to_process(to_process, batch.batch, batch.edge_index[0])
            mask_cp = to_process.bool()
        else:
            mask, edge_mask = utils.get_mask_to_process(torch.sigmoid(continue_p), batch.batch, batch.edge_index[0])
            mask_cp = (continue_p > 0.0).bool()
            if enforced_mask is not None:
                enforced_mask_ids = enforced_mask[batch.batch]
                enforced_mask_edge_ids = enforced_mask_ids[batch.edge_index[0]]
                mask &= enforced_mask_ids
                mask_cp &= enforced_mask
                edge_mask &= enforced_mask_edge_ids
        return mask, mask_cp, edge_mask

    def __init__(self, latent_features, node_features, edge_features, output_features, algo_processor, dataset_class, dataset_root, bias=False):
        super(AlgorithmBase, self).__init__()
        self.dataset_class = dataset_class
        self.dataset_root = dataset_root
        self.train_dataset = dataset_class(dataset_root, split='train', less_wired=True, device='cpu')
        self.val_dataset = dataset_class(dataset_root, split='val', less_wired=True, device='cpu')

        self.processor = algo_processor.processor
        self.decoder_network = DecoderNetwork(2 * latent_features, output_features, bias=bias)

        self.termination_network = nn.Sequential(
            nn.Linear(latent_features, 1, bias=bias),
        )
        def printer(module, gradInp, gradOutp): # Debug for backward hook
            s=0
            mx=-float('inf')
            for gi in gradInp:
                s += torch.sum(gi)
                mx = max(mx, torch.max(gi))

            print(s, mx)
            s=0
            mx=-float('inf')
            for go in gradOutp:
                s += torch.sum(go)
                mx = max(mx, torch.max(go))
            print(s, mx)

    def get_continue_p(self, batch_ids, latent_nodes, GRAPH_SIZES):
        graph_latent = utils.get_graph_embedding(batch_ids, latent_nodes, GRAPH_SIZES)
        continue_p = self.termination_network(graph_latent).view(-1)
        return continue_p

    def update_termination_statistics(self, continue_p_masked, true_termination_masked):
        for i, cp in enumerate(continue_p_masked):
            if torch.sigmoid(cp) > 0.5:
                if true_termination_masked[i] == 1:
                    self.true_positive += 1
                if true_termination_masked[i] == 0:
                    self.false_positive += 1
            else:
                if true_termination_masked[i] == 1:
                    self.false_negative += 1
                if true_termination_masked[i] == 0:
                    self.true_negative += 1

    @staticmethod
    def calculate_step_acc(output, output_real):
        """ Calculates the accuracy for a givens step """
        correct = 0
        tot = 0
        correct = (output == output_real).sum(dim=1)
        non_inf = output_real != -1
        tot = non_inf.sum(dim=1)
        return (correct, tot)

    def zero_steps(self):
        self.sum_of_processed_nodes, self.steps, self.sum_of_steps, self.cnt = 0, 0, 0, 0

    def zero_termination(self):
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.true_negative = 0

    def aggregate_steps(self, steps, processed_nodes):
        self.sum_of_steps += steps
        self.sum_of_processed_nodes += processed_nodes
        self.steps += 1
        if not self.training:
            self.validation_sum_of_processed_nodes += processed_nodes
            self.validation_sum_of_steps += steps

    def set_initial_last_states(self, batch, STEPS_SIZE, SOURCE_NODES):
        hyperparameters = get_hyperparameters()
        DEVICE = hyperparameters["device"]
        DIM_LATENT = hyperparameters["dim_latent"]

        SIZE = batch.num_nodes
        self.last_latent = torch.zeros(SIZE, DIM_LATENT, device=DEVICE)
        self.last_continue_p = torch.ones(batch.num_graphs, device=DEVICE)
        x, y = self.get_input_output_features(batch, SOURCE_NODES)
        x.requires_grad = False
        y.requires_grad = False
        x_curr, _ = self.get_step_io(x, y)
        self.last_output = x_curr[:, 0].clone()
    
    def update_states(self, continue_p, current_latent):
        DIM_LATENT = get_hyperparameters()["dim_latent"]
        self.last_continue_p = torch.where(self.mask_cp, continue_p, self.last_continue_p)
        self.last_latent = torch.where(self.mask.unsqueeze(1).repeat_interleave(DIM_LATENT, dim=1), current_latent, self.last_latent)
        return self.last_continue_p

    def get_step_io(self, x, y):
        DEVICE = get_hyperparameters()["device"]
        x_curr = torch.index_select(x, 1, torch.tensor([self.steps], dtype=torch.long, device=DEVICE)).squeeze(1)
        y_curr = torch.index_select(y, 1, torch.tensor([self.steps], dtype=torch.long, device=DEVICE)).squeeze(1)
        return x_curr, y_curr

    def prepare_constants(self, batch):
        SIZE = batch.num_nodes
        # we make at most |V|-1 steps
        GRAPH_SIZES, SOURCE_NODES, SINK_NODES = utils.get_sizes_and_source_sink(batch)
        STEPS_SIZE = GRAPH_SIZES.max()-1
        return SIZE, GRAPH_SIZES, SOURCE_NODES, STEPS_SIZE, SINK_NODES

    def prepare_initial_masks(self, batch):
        DEVICE = get_hyperparameters()["device"]
        mask = torch.ones_like(batch.batch, dtype=torch.bool, device=DEVICE)
        mask_cp = torch.ones(batch.num_graphs, dtype=torch.bool, device=DEVICE)
        edge_mask = torch.ones_like(batch.edge_index[0], dtype=torch.bool, device=DEVICE)
        return mask, mask_cp, edge_mask

    def loop_condition(self, batch_ids, x, y, STEPS_SIZE, GRAPH_SIZES):
        return (((not self.training and self.mask_cp.any()) or
                 (self.training and utils.finish(x, y, batch_ids, self.steps, STEPS_SIZE, GRAPH_SIZES).bool().any())) and
                 self.steps < STEPS_SIZE and
                 not utils.interrupted())

    def get_training_loss(self):
        losses =\
                type(self).get_losses_from_predictions(self.predictions, self.actual)
        if not self.steps:
            return 0
        return sum(losses)

    def get_losses_dict(self):
        return self.losses

    def get_validation_losses(self):
        if not self.steps:
            return tuple(0 for _ in range(len(self.validation_predictions)))
        return type(self).get_losses_from_predictions(self.validation_predictions, self.validation_actual)


    def get_validation_accuracies(self):
        return sum(self.mean_step)/len(self.mean_step), sum(self.last_step)/self.last_step_total.float()

    def process(
            self,
            batch,
            EPSILON=1,
            enforced_mask=None,
            compute_losses_and_broken=True,
            debug=False):

        DEVICE = get_hyperparameters()["device"]
        train = self.training
        SIZE, GRAPH_SIZES, SOURCE_NODES, STEPS_SIZE, SINK_NODES = self.prepare_constants(batch)

        batch = utils.add_self_loops(batch) # also takes into account capacities/weights
        self.zero_tracking_losses_and_statistics()
        self.set_initial_last_states(batch, STEPS_SIZE, SOURCE_NODES)
        adj_matrix, flow_matrix = utils.get_adj_flow_matrix(SIZE, batch.edge_index, batch.edge_attr[:, 1])
        x, y = self.get_input_output_features(batch, SOURCE_NODES)
        self.mask, self.mask_cp, self.edge_mask = self.prepare_initial_masks(batch)
        assert self.mask_cp.all(), self.mask_cp
        self.processor.zero_hidden(batch.num_nodes)

        while self.loop_condition(batch.batch, x, y, STEPS_SIZE, GRAPH_SIZES):

            self.x_curr, self.y_curr = self.get_step_io(x, y)
            assert self.mask_cp.any(), self.mask_cp
            if not self.training:
                assert (self.last_continue_p > 0).any()
            start = time.time()
            to_process = utils.finish(x, y, batch.batch, self.steps, STEPS_SIZE, GRAPH_SIZES).bool()
            true_termination = utils.finish(x, y, batch.batch, self.steps+1, STEPS_SIZE, GRAPH_SIZES)
            assert self.mask_cp.any(), to_process if self.training else self.last_continue_p
            inp = utils.get_input(batch, EPSILON, train, self.x_curr, self.last_output)

            start = time.time()
            self.loop_body(batch, inp, true_termination, to_process, compute_losses_and_broken, enforced_mask, GRAPH_SIZES)
            self.mask, self.mask_cp, self.edge_mask = type(self).get_masks(self.training, batch, to_process, self.last_continue_p, enforced_mask)

        outputs = self.get_outputs(batch, adj_matrix, flow_matrix, compute_losses_and_broken)
        if type(self) == models.AugmentingPathNetwork:
            walks, mask_end_of_path = utils.get_walks(self.training, batch, outputs, GRAPH_SIZES, SOURCE_NODES, SINK_NODES)
            mins = self.find_mins(batch, walks, mask_end_of_path, GRAPH_SIZES, SOURCE_NODES, SINK_NODES)
            flows = self.augment_flow(batch, walks, mask_end_of_path, mins)

        batch.edge_index, batch.edge_attr = torch_geometric.utils.remove_self_loops(batch.edge_index, batch.edge_attr)
        return outputs

    def zero_tracking_losses_and_statistics(self):
        self.zero_steps()
        if self.training:
            self.zero_termination()

    def zero_validation_stats(self):
        self.mean_step = []
        self.last_step = []
        self.validation_sum_of_steps = 0
        self.validation_sum_of_processed_nodes = 0
        self.last_step_total = 0
        self.zero_termination()

    def aggregate_last_step(self, output, real): 
        correct, tot = AlgorithmBase.calculate_step_acc(output.unsqueeze(0), real.unsqueeze(0))
        self.last_step.append(correct.squeeze())
        self.last_step_total += tot.squeeze()

