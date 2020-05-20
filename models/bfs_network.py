import torch
import torch.nn as nn
import torch.nn.functional as F

from overrides import overrides

import utils
from layers import DecoderNetwork
from models import AlgorithmBase
from hyperparameters import get_hyperparameters

class BFSNetwork(AlgorithmBase):
    def __init__(self, latent_features, node_features, edge_features, algo_processor, dataset_class, dataset_root, bias=False):
        super(BFSNetwork, self).__init__(latent_features, node_features, edge_features, 1, algo_processor, dataset_class, dataset_root, bias=bias)

        self.node_encoder = nn.Sequential(
            nn.Linear(node_features + latent_features, latent_features, bias=bias),
            nn.LeakyReLU()
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, latent_features, bias=bias),
            nn.LeakyReLU()
        )

    def zero_tracking_losses_and_statistics(self):
        super().zero_tracking_losses_and_statistics()
        self.losses = {
            "total_loss_reachability": 0,
            "total_loss_term": 0,
        }
        self.predictions = {
            "reachabilities": [],
            "terminations": [],
        }
        self.actual = {
            "reachabilities": [],
            "terminations": [],
        }

    def get_step_loss(self,
                      batch_ids, mask, mask_cp,
                      y_curr, continue_p, true_termination,
                      reachable_p,
                      compute_losses_and_broken=True):
        reachable_p_masked = reachable_p[mask]
        reachable_real_masked = y_curr[mask].squeeze()
        steps = sum(mask_cp.float())

        loss_reachable, loss_term, processed_nodes, step_acc = 0, 0, 0, 1

        if compute_losses_and_broken:
            processed_nodes = len(reachable_real_masked)
            self.predictions["reachabilities"].extend(reachable_p_masked)
            self.actual["reachabilities"].extend(reachable_real_masked)
            self.predictions["terminations"].extend(continue_p[mask_cp])
            self.actual["terminations"].extend(true_termination[mask_cp])
            loss_reachable = F.binary_cross_entropy_with_logits(reachable_p_masked, reachable_real_masked, reduction='sum', pos_weight=torch.tensor(1.00))
            loss_term = F.binary_cross_entropy_with_logits(continue_p[mask_cp], true_termination[mask_cp], reduction='sum', pos_weight=torch.tensor(1.00))
            if get_hyperparameters()["calculate_termination_statistics"]:
                self.update_termination_statistics(continue_p[mask_cp], true_termination[mask_cp])
            
            if not self.training:
                reachable_split = utils.split_per_graph(batch_ids, reachable_p > 0)
                reachable_real_split = utils.split_per_graph(batch_ids, y_curr.squeeze())
                correct, tot = BFSNetwork.calculate_step_acc(reachable_split[mask_cp], reachable_real_split[mask_cp])
                self.mean_step.extend(correct/tot.float())
                step_acc = correct/tot.float()

        return loss_reachable, loss_term, steps, processed_nodes, step_acc

    def get_input_output_features(self, batch, SOURCE_NODES): #Ignored parameter in BFS
        x = batch.x.clone().unsqueeze(2)
        y = batch.y.clone().unsqueeze(2)
        return x, y

    def aggregate_loss_steps_and_acc(
            self,
            batch_ids, mask, mask_cp,
            continue_p, true_termination,
            compute_losses_and_broken,
            y_curr, reachable_p):

        loss_reachable, loss_term, steps, processed_nodes, step_acc =\
                self.get_step_loss(
                    batch_ids, mask, mask_cp,
                    y_curr,
                    continue_p, true_termination,
                    reachable_p,
                    compute_losses_and_broken=compute_losses_and_broken)

        self.losses["total_loss_reachability"] += loss_reachable
        self.losses["total_loss_term"] += loss_term
        self.aggregate_steps(steps, processed_nodes)

    def set_initial_last_states(self, batch, STEPS_SIZE, SOURCE_NODES):
        hyperparameters = get_hyperparameters()
        DEVICE = hyperparameters["device"]
        DIM_LATENT = hyperparameters["dim_latent"]

        SIZE = batch.num_nodes
        INF = STEPS_SIZE
        super().set_initial_last_states(batch, STEPS_SIZE, SOURCE_NODES)
        self.last_reachable_p = torch.full([SIZE], -1e3, device=DEVICE)
        self.last_reachable_p[0] = 1e3
        self.last_reachable = torch.zeros(SIZE)
        self.last_reachable[0] = 1.

    def update_states(self, reachable_p, continue_p, current_latent):
        super().update_states(continue_p, current_latent)
        DIM_LATENT = get_hyperparameters()["dim_latent"]
        self.last_reachable_p = torch.where(self.mask, reachable_p, self.last_reachable_p)
        self.last_reachable = (self.last_reachable_p >= 0.5).float()
        self.last_output = torch.where(self.mask, self.last_reachable, self.last_output)

    def loop_body(
        self, batch, inp, true_termination, to_process,
        compute_losses_and_broken, enforced_mask, GRAPH_SIZES
    ):
        current_latent, reachable_p, continue_p =\
            self(
                batch.batch,
                GRAPH_SIZES,
                inp,
                self.last_latent,
                batch.edge_index,
                batch.edge_attr
            )
        self.update_states(reachable_p, continue_p, current_latent)

        self.aggregate_loss_steps_and_acc(
            batch.batch, self.mask, self.mask_cp,
            self.last_continue_p, true_termination,
            compute_losses_and_broken,
            self.y_curr, self.last_reachable_p)


    def get_outputs(self, batch, adj_matrix, flow_matrix, compute_losses_and_broken):# Also updates broken invariants
        reachable = self.last_reachable_p >= 0.5
        return reachable

    @overrides
    def get_losses_dict(self):
        return {
            "total_loss_reachability": self.losses["total_loss_reachability"] / self.sum_of_processed_nodes,
            "total_loss_term": self.losses["total_loss_term"] / self.sum_of_steps
        }

    @overrides
    def get_training_loss(self):
        return sum(self.get_losses_dict().values())

    @overrides
    def get_validation_losses(self):
        total_loss_reachability = self.validation_losses["total_loss_reachability"] / float(self.validation_sum_of_processed_nodes)
        total_loss_term = self.validation_losses["total_loss_term"] / float(self.validation_sum_of_steps)
        return total_loss_reachability, total_loss_term

    @staticmethod
    def get_losses_from_predictions(predictions, actual):
        for key in predictions:
            if not isinstance(predictions[key], list):
                continue
            predictions[key] = torch.stack(predictions[key], dim=0)
            actual[key] = torch.stack(actual[key], dim=0)

        total_loss_reachability = F.binary_cross_entropy_with_logits(predictions["reachabilities"], actual["reachabilities"])
        total_loss_term = F.binary_cross_entropy_with_logits(predictions["terminations"], actual["terminations"])
        return total_loss_reachability, total_loss_term

    def zero_validation_stats(self):
        super().zero_validation_stats()
        self.validation_losses = {
            "total_loss_reachability": 0,
            "total_loss_term": 0,
        }
        self.validation_predictions = {
            "reachabilities": [],
            "terminations": [],
        }
        self.validation_actual = {
            "reachabilities": [],
            "terminations": [],
        }

    def update_validation_stats(self, batch, reachabilities):
        _, y = self.get_input_output_features(batch, None)
        reachabilities_real = y[:, -1]
        super().aggregate_last_step(reachabilities, reachabilities_real.squeeze())
        for key in self.validation_losses:
            self.validation_losses[key] += self.losses[key]
        for key in self.validation_predictions:
            self.validation_predictions[key].extend(self.predictions[key])
            self.validation_actual[key].extend(self.actual[key])

    def forward(self, batch_ids, GRAPH_SIZES, current_input, last_latent, edge_index, edge_attr, edge_mask=None):
        SIZE = last_latent.shape[0]

        inp = torch.cat((current_input.unsqueeze(1), last_latent), dim=1)
        encoded_nodes = self.node_encoder(inp)
        encoded_edges = self.edge_encoder(edge_attr[:, 1].unsqueeze(1))
        latent_nodes = self.processor(encoded_nodes, encoded_edges, utils.flip_edge_index(edge_index))
        output = self.decoder_network(torch.cat((encoded_nodes, latent_nodes), dim=1))

        continue_p = self.get_continue_p(batch_ids, latent_nodes, GRAPH_SIZES)
        return latent_nodes, output.squeeze(), continue_p
