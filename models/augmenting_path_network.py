import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_cluster import random_walk

from hyperparameters import get_hyperparameters
from half_deterministic import obtain_paths, get_pairs
import utils
from layers import PredecessorNetwork, GAT
from models import AlgorithmBase
from overrides import overrides

class AugmentingPathNetwork(AlgorithmBase):
    def __init__(self, latent_features, node_features, edge_features, algo_processor, dataset_class, dataset_root, bias=False, use_ints=False, bits_size=None):
        super(AugmentingPathNetwork, self).__init__(latent_features, node_features, edge_features, bits_size if use_ints else 1, algo_processor, dataset_class, dataset_root, bias=bias)
        self.bits_size = bits_size
        if use_ints:
            self.bit_encoder = nn.Sequential(
                nn.Linear(bits_size, latent_features, bias=bias),
                nn.LeakyReLU()
            )

        ne_input_features = 2*latent_features if use_ints else node_features+latent_features
        self.node_encoder = nn.Sequential(
            nn.Linear(ne_input_features, latent_features, bias=bias),
            nn.LeakyReLU()
        )

        ee_input_features = 2*latent_features if use_ints else edge_features
        self.edge_encoder = nn.Sequential(
            nn.Linear(ee_input_features, latent_features, bias=bias),
            nn.LeakyReLU()
        )

        self.pred_network = PredecessorNetwork(latent_features, latent_features, bias=bias)
        if not use_ints:
            self.infinity = nn.Parameter(torch.randn(latent_features))

        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=latent_features, dim_feedforward=32, nhead=4)
        self.transformer_enc_lin = nn.Linear(latent_features, 1)
        self.subtract_network = nn.Linear(2*latent_features, (2**bits_size))

        def printer(module, gradInp, gradOutp): # Printer for backward hook
            print("GI", gradInp)
            print("GO", gradOutp)
            s=0
            mx=-float('inf')
            for gi in gradInp:
                s += torch.sum(gi)
                mx = max(mx, torch.max(gi))

            print("GISUMAX", s, mx)
            s=0
            mx=-float('inf')
            for go in gradOutp:
                s += torch.sum(go)
                mx = max(mx, torch.max(go))
                print("GOSUMdim1", go.sum(dim=1))
            print("GOSUMAX", s, mx)
            input()

    @staticmethod
    def get_real_output_values(y_curr):
        DEVICE = 'cuda' if y_curr.is_cuda else 'cpu'
        zero_selector = torch.tensor([0], dtype=torch.long, device=DEVICE)
        one_selector = torch.tensor([1], dtype=torch.long, device=DEVICE)
        distances_real = torch.index_select(y_curr, 1, zero_selector).squeeze()
        predecessors_real = torch.index_select(y_curr, 1, one_selector).long().squeeze()
        return distances_real, predecessors_real

    @staticmethod
    def mask_infinities(mask, y_curr, distances, predecessors_p):
        distances_real, predecessors_real = AugmentingPathNetwork.get_real_output_values(y_curr)

        non_inf_indices = (predecessors_real[mask] != -1).nonzero().squeeze()
        distances = torch.index_select(distances[mask], 0, non_inf_indices)
        predecessors_p_masked = torch.index_select(predecessors_p[mask], 0, non_inf_indices)
        distances_real = torch.index_select(distances_real[mask], 0, non_inf_indices)
        predecessors_real = torch.index_select(predecessors_real[mask], 0, non_inf_indices)
        return distances, distances_real, predecessors_p_masked, predecessors_real

    def zero_tracking_losses_and_statistics(self):
        super().zero_tracking_losses_and_statistics()
        self.losses = {
            "pred": 0,
            "dist": 0,
            "term": 0,
            "minimum": 0,
            "augment": 0
        }


    def get_step_loss(self,
                      batch_ids, mask, mask_cp,
                      y_curr,
                      continue_p, true_termination,
                      distances, predecessors_p,
                      compute_losses_and_broken=True):
        distances_masked, distances_real_masked, predecessors_p_masked, predecessors_real_masked = \
                AugmentingPathNetwork.mask_infinities(mask, y_curr, distances, predecessors_p)
        steps = sum(mask_cp.float())

        train = self.training

        loss_dist, loss_pred, loss_term, processed_nodes, step_acc = 0, 0, 0, 0, 1
        if distances_real_masked.nelement() != 0 and compute_losses_and_broken:
            processed_nodes = len(distances_real_masked)
            if self.bits_size is None:
                loss_dist = F.mse_loss(distances_masked, distances_real_masked, reduction='sum')
            else:
                loss_dist = F.binary_cross_entropy_with_logits(distances_masked, utils.integer2bit(distances_real_masked), reduction='sum')
            loss_pred = F.cross_entropy(predecessors_p_masked, predecessors_real_masked, ignore_index=-1, reduction='sum')

        if compute_losses_and_broken:
            assert mask_cp.any(), mask_cp
            loss_term = F.binary_cross_entropy_with_logits(continue_p[mask_cp], true_termination[mask_cp], reduction='sum', pos_weight=torch.tensor(1.00))
            if get_hyperparameters()["calculate_termination_statistics"]:
                self.update_termination_statistics(continue_p[mask_cp], true_termination[mask_cp])

            assert loss_term.item() != float('inf')

        if not train and mask_cp.any() and compute_losses_and_broken:
            assert mask_cp.any()
            _, predecessors_real = AugmentingPathNetwork.get_real_output_values(y_curr)
            predecessors_p_split = utils.split_per_graph(batch_ids, predecessors_p)
            predecessors_real_split = utils.split_per_graph(batch_ids, predecessors_real)
            correct, tot = AugmentingPathNetwork.calculate_step_acc(torch.max(predecessors_p_split[mask_cp], dim=2).indices, predecessors_real_split[mask_cp])
            self.mean_step.extend(correct/tot.float())
            step_acc = correct/tot.float()

        return loss_dist, loss_pred, loss_term, steps, processed_nodes, step_acc

    @staticmethod
    def get_input_infinity_mask(inp):
        mask = inp[:, :, 1] == -1
        return mask


    def get_input_output_features(self, batch, SOURCE_NODES):
        x = batch.x.clone()
        y = batch.y.clone()
        mask_x = AugmentingPathNetwork.get_input_infinity_mask(batch.x)
        mask_y = AugmentingPathNetwork.get_input_infinity_mask(batch.y)
        x[:, :, 1] += SOURCE_NODES[batch.batch].unsqueeze(1)
        y[:, :, 1] += SOURCE_NODES[batch.batch].unsqueeze(1)
        x[:, :, 1][mask_x] = -1
        y[:, :, 1][mask_y] = -1
        return x, y

    def aggregate_loss_steps_and_acc(
            self,
            batch_ids, mask, mask_cp,
            compute_losses_and_broken,
            continue_p, true_termination,
            y_curr,
            distances, predecessors_p):
        loss_dist, loss_pred, loss_term, steps, processed_nodes, step_acc =\
            self.get_step_loss(
                batch_ids, mask, mask_cp,
                y_curr,
                continue_p, true_termination,
                distances, predecessors_p,
                compute_losses_and_broken=compute_losses_and_broken)

        self.losses["dist"] += loss_dist
        self.losses["pred"] += loss_pred
        self.losses["term"] += loss_term
        self.aggregate_steps(steps, processed_nodes)

    def set_initial_last_states(self, batch, STEPS_SIZE, SOURCE_NODES): 
        hyperparameters = get_hyperparameters()
        DEVICE = hyperparameters["device"]
        DIM_LATENT = hyperparameters["dim_latent"]
        DIM_NODES = hyperparameters["dim_nodes_AugmentingPath"]
        DIM_EDGES = hyperparameters["dim_edges"]

        SIZE = batch.num_nodes
        super().set_initial_last_states(batch, STEPS_SIZE, SOURCE_NODES)
        self.last_predecessors_p = torch.full((SIZE, SIZE), -1e9, device=DEVICE)
        self.last_predecessors_p[(SOURCE_NODES, SOURCE_NODES)] = 1e9
        self.last_distances = self.last_output.clone()
        self.last_distances[SOURCE_NODES] = 0.

    def update_states(self, distances, predecessors_p,
                      continue_p, current_latent):
        super().update_states(continue_p, current_latent)
        DIM_LATENT = get_hyperparameters()["dim_latent"]
        self.last_distances = torch.where(self.mask, utils.bit2integer(distances), self.last_distances)
        self.last_predecessors_p = torch.where(self.mask, predecessors_p, self.last_predecessors_p)
        self.last_output = self.last_distances

    
    def loop_body(
            self, batch, inp, true_termination, to_process,
            compute_losses_and_broken, enforced_mask, GRAPH_SIZES
    ):
        assert not self.training or to_process.any()
        assert self.mask_cp.any()
        iimask = AugmentingPathNetwork.get_input_infinity_mask(batch.x)[:, 0]
        current_latent, distances, predecessors_p, continue_p = \
            self(
                batch.batch,
                GRAPH_SIZES,
                inp,
                self.last_latent,
                batch.edge_index,
                batch.edge_attr,
                iimask
            )
        self.update_states(distances, predecessors_p,
                           continue_p, current_latent)
        start = time.time()

        self.aggregate_loss_steps_and_acc(
            batch.batch, self.mask, self.mask_cp,
            compute_losses_and_broken,
            self.last_continue_p, true_termination,
            self.y_curr,
            distances, predecessors_p)

    @overrides
    def get_training_loss(self):
        return sum(self.get_losses_dict().values())

    @overrides
    def get_validation_losses(self):
        denom = self.validation_sum_of_processed_nodes
        if self.bits_size is not None:
            denom *= self.bits_size
        dist = self.validation_losses["dist"] / float(denom) if self.sum_of_steps != 0 else 0
        pred = self.validation_losses["pred"] / float(self.validation_sum_of_processed_nodes) if self.sum_of_steps != 0 else 0
        term = self.validation_losses["term"] / float(self.validation_sum_of_steps) if self.sum_of_steps != 0 else 0
        minimum = self.validation_losses["minimum"] / math.ceil(len(self.val_dataset) / float(get_hyperparameters()["batch_size"])) if self.sum_of_steps != 0 else 0
        return dist, pred, term, minimum

    @overrides
    def get_validation_accuracies(self):
        return (sum(self.mean_step)/len(self.mean_step),
                sum(self.last_step)/self.last_step_total.float(),
                self.mincaps_TN.float()/(self.mincaps_FP.float()+
                                         self.mincaps_TN.float()),
                self.subtract_correct/float(self.subtract_all))

    @overrides
    def get_losses_dict(self):
        denom = self.sum_of_processed_nodes
        if self.bits_size is not None:
            denom *= self.bits_size
        return {
            "dist": self.losses["dist"] / float(denom) if self.sum_of_steps != 0 else 0,
            "pred": self.losses["pred"] / self.sum_of_processed_nodes if self.sum_of_steps != 0 else 0,
            "term": self.losses["term"] / self.sum_of_steps if self.sum_of_steps != 0 else 0,
            "minimum": self.losses["minimum"] / 10.0,
            "augment": self.losses["augment"] / 10.0
        }

    def zero_validation_stats(self):
        super().zero_validation_stats()
        self.broken_invariants = []
        self.broken_flows = []
        self.broken_reachabilities = []
        self.broken_all = []
        self.mincaps_TN = 0
        self.mincaps_FP = 0
        self.subtract_correct = 0
        self.subtract_all = 0
        self.validation_losses = {
            "pred": 0,
            "dist": 0,
            "term": 0,
            "minimum": 0,
        }

    def update_broken_invariants(self, batch, predecessors, adj_matrix, flow_matrix):

        start = time.time()
        DEVICE = get_hyperparameters()["device"]
        GRAPH_SIZES, SOURCE_NODES, SINK_NODES = utils.get_sizes_and_source_sink(batch)
        STEPS_SIZE = GRAPH_SIZES.max()
        _, y = self.get_input_output_features(batch, SOURCE_NODES)
        broke_flow = torch.zeros(batch.num_graphs, dtype=torch.bool, device=DEVICE)
        broke_reachability_source = torch.zeros(batch.num_graphs, dtype=torch.bool, device=DEVICE)
        broke_invariant = torch.zeros(batch.num_graphs, dtype=torch.bool, device=DEVICE)
        curr_node = SINK_NODES.clone().detach()
        cnt = 0
        predecessors_real = y[:, -1, -1]

        idx = predecessors[curr_node] != curr_node
    
        while (predecessors_real[SINK_NODES] != -1).any() and cnt <= STEPS_SIZE and idx.any() and not utils.interrupted():
            # Ignore if we reached the starting node loop
            # (predecessor[starting node] = starting node)
            move_to_predecessors = torch.stack((predecessors[curr_node], curr_node), dim=0)[:, idx]
            rowcols = (move_to_predecessors[0], move_to_predecessors[1])
            if not adj_matrix[rowcols].all():
                # each predecessor lead to a node accessible by an edge!!!
                print()
                print(adj_matrix)
                print(curr_node)
                print(predecessors[curr_node])
                print("FATAL INVARIANT ERORR")
                exit(0)

            assert adj_matrix[rowcols].all()

            if (flow_matrix[rowcols] <= 0).any():
                broke_flow[idx] |= flow_matrix[rowcols] <= 0
            curr_node[idx] = predecessors[curr_node[idx]]
            idx = (predecessors[curr_node] != curr_node) & (predecessors_real[SINK_NODES] != -1)
            cnt += 1
            if cnt > STEPS_SIZE+1:
                break

        original_reachable_mask = (predecessors_real[SINK_NODES] != -1)
        broke_reachability_source |= (curr_node != SOURCE_NODES)
        broke_invariant = broke_flow | broke_reachability_source
        broke_all = broke_flow & broke_reachability_source
        
        self.broken_invariants.extend((original_reachable_mask & broke_invariant).clone().detach())
        self.broken_reachabilities.extend((original_reachable_mask & broke_reachability_source).clone().detach())
        self.broken_flows.extend((original_reachable_mask & broke_flow).clone().detach())
        self.broken_all.extend((original_reachable_mask & broke_all).clone().detach())

    def get_outputs(self, batch, adj_matrix, flow_matrix, compute_losses_and_broken):# Also updates broken invariants
        predecessors = torch.max(self.last_predecessors_p, dim=1).indices
        if not self.training and compute_losses_and_broken:
            self.update_broken_invariants(batch, predecessors, adj_matrix, flow_matrix)
        return predecessors

    def get_broken_invariants(self):
        return self.broken_invariants, self.broken_reachabilities, self.broken_flows, self.broken_all

    def update_validation_stats(self, batch, predecessors):
        _, SOURCE_NODES, _ = utils.get_sizes_and_source_sink(batch)
        _, y = self.get_input_output_features(batch, SOURCE_NODES)
        predecessors_real = y[:, -1, -1]

        super().aggregate_last_step(predecessors, predecessors_real)
        for key in self.validation_losses:
            self.validation_losses[key] += self.losses[key]

    def encode_edges(self, edge_attr):
        if self.bits_size is not None:
            edge_attr_w = self.bit_encoder(utils.integer2bit(edge_attr[:, 0]))
            edge_attr_cap = self.bit_encoder(utils.integer2bit(edge_attr[:, 1]))
            edge_attr = torch.cat((edge_attr_w, edge_attr_cap), dim=1)
        encoded_edges = self.edge_encoder(edge_attr)
        return encoded_edges

    def get_features_from_walk(self, walks, batch, mask_end_of_path=None):
        attr_matrix = torch_geometric.utils.to_dense_adj(batch.edge_index, edge_attr=batch.edge_attr).squeeze()
        walk_nodes_latent = self.last_latent[walks]
        walk_nodes_latent_i = walk_nodes_latent[:, :-1]
        walk_nodes_latent_j = walk_nodes_latent[:, 1:]
        walk_nodes_latent_i = walk_nodes_latent_i.reshape(-1, get_hyperparameters()["dim_latent"])
        walk_nodes_latent_j = walk_nodes_latent_j.reshape(-1, get_hyperparameters()["dim_latent"])
        wf = walks[:, :-1]
        ws = walks[:, 1:]
        walk_edge_index = torch.stack((wf, ws), dim=0)
        walk_edge_index = walk_edge_index.view(2,-1)

        walk_attrs = attr_matrix[(ws, wf)]
        inv_walk_attrs = attr_matrix[(wf, ws)]
        if self.training:
            walk_attrs[:, :, 1] = walk_attrs[:, :, 1] + torch.randint_like(walk_attrs[:, :, 1], low=0, high=10)
        wa = walk_attrs[:, :, 1].clone().detach()
        if mask_end_of_path is not None:
            wa[mask_end_of_path[:, 1:]] = 1e9
        actual_argmins = wa.argmin(dim=1)
        no_step_mask = (wa.min(dim=1).values == 1e9) # no step was made by the algorithm so min is not defined
        return walk_nodes_latent_i, walk_nodes_latent_j, walk_edge_index, walk_attrs, inv_walk_attrs, actual_argmins, no_step_mask

    def get_messages_from_features(self, x_i, x_j, walk_edge_index, attrs, batch):
        attrs = attrs.reshape(-1, get_hyperparameters()["dim_edges"])
        enc_attrs = self.encode_edges(attrs)
        messages = (
                self.processor.message(x_i, x_j, utils.flip_edge_index(walk_edge_index), enc_attrs, batch.num_nodes)
                if type(self.processor) == GAT else
                self.processor.message(x_i, x_j, enc_attrs))
        messages = messages.reshape(batch.num_graphs, -1, get_hyperparameters()["dim_latent"])
        return messages

    def augment_flow(self, batch, walks, mask_end_of_path, mins):
        BATCH_SIZE = batch.num_graphs
        x_i, x_j, walk_edge_index, attrs, inv_walk_attrs, actual_argmins, no_step_mask =\
                self.get_features_from_walk(walks, batch, mask_end_of_path=mask_end_of_path)
        if no_step_mask.all():
            return attrs[:, :, 1] # nothing changes

        if self.training: attrs = self._attrs #Augmentation and bottleneck finding had to have the same edge attributes
        messages = self.get_messages_from_features(x_i, x_j, walk_edge_index, attrs, batch)
        messages_inv = self.get_messages_from_features(x_j, x_i, walk_edge_index, inv_walk_attrs, batch)
        minemb = self.bit_encoder(utils.integer2bit(mins.float()))
        minemb = minemb.unsqueeze(1).repeat_interleave(messages.shape[1], dim=1)
        new_weight_distribution = self.subtract_network(torch.cat((minemb, messages), dim=-1))

        mask = torch.arange(2**self.bits_size, device=get_hyperparameters()["device"])
        mask = mask.repeat(BATCH_SIZE, attrs.shape[1], 1)
        expanded_caps = attrs[:, :, 1].unsqueeze(2).expand_as(mask)
        mask = mask > expanded_caps
        new_weight_distribution[mask] = -1e9
        new_weight_distribution = new_weight_distribution[~no_step_mask]
        real_new_caps = attrs[:, :, 1][~no_step_mask]
        real_new_caps -= mins[~no_step_mask].unsqueeze(-1)
        if mask_end_of_path is not None:
            real_new_caps[mask_end_of_path[:, 1:][~no_step_mask]] = -100 # ignore value of cross entropy
        just_correct = real_new_caps >= 0 # the neural mins provided may be incorrect so ignore these values
        neural_new_caps = new_weight_distribution.argmax(dim=-1)
        if just_correct.any():
            self.losses["augment"] = F.cross_entropy(new_weight_distribution[just_correct].view(-1, 2**self.bits_size), real_new_caps[just_correct].view(-1).long())
            subtract_correct = (neural_new_caps == real_new_caps)[just_correct].sum()
            subtract_all = real_new_caps[just_correct].nelement()
            if not self.training:
                self.subtract_correct += subtract_correct
                self.subtract_all += subtract_all
        else:
            return attrs[:, :, 1] # nothing changes
        return neural_new_caps

    def find_mins(self, batch, walks, mask_end_of_path, GRAPH_SIZES, SOURCE_NODES, SINK_NODES):
        def pass_through_min_net(x_i, x_j, walk_edge_index, attrs, actual_argmins, mask=None, no_step_mask=None):
            messages = self.get_messages_from_features(x_i, x_j, walk_edge_index, attrs, batch)
            if not self.training:
                mask = mask[:, 1:]
            minimums = self.transformer_encoder(messages.permute(1, 0, 2), src_key_padding_mask=mask)
            minimums = self.transformer_enc_lin(minimums).permute(1, 0, 2).squeeze(2)
            if mask is not None:
                minimums[mask] = float('-inf')
            if no_step_mask is None:
                no_step_mask = torch.zeros_like(actual_argmins, dtype=torch.bool)

            if (~no_step_mask).any():
                self.losses["minimum"] = F.cross_entropy(minimums[~no_step_mask], actual_argmins[~no_step_mask])
                if math.isnan(self.losses["minimum"].item()) or math.isinf(self.losses["minimum"].item()):
                    print("NO STEP MASK", no_step_mask)
                    print("MINIMUMS", minimums)
                    print("actual_argmins", actual_argmins)
                    print("M[AM]", minimums[(torch.arange(len(minimums)), actual_argmins)])
                    input()
            return minimums

        x_i, x_j, walk_edge_index, attrs, _, actual_argmins, no_step_mask = self.get_features_from_walk(walks, batch, mask_end_of_path=mask_end_of_path)
        self._attrs = attrs
        mins = pass_through_min_net(x_i, x_j, walk_edge_index, attrs, actual_argmins, mask=mask_end_of_path, no_step_mask=no_step_mask)
        actual_mincaps = attrs[torch.arange(batch.num_graphs), actual_argmins, 1]
        if self.training:
            return actual_mincaps
        argmins = mins.argmax(dim=1) # yes, mins are logits for the min position, hence argmax
        mincaps = attrs[torch.arange(batch.num_graphs), argmins, 1]
        TN = torch_geometric.utils.true_negative(mincaps[~no_step_mask], actual_mincaps[~no_step_mask], 2).sum()
        FP = torch_geometric.utils.false_positive(mincaps[~no_step_mask], actual_mincaps[~no_step_mask], 2).sum()
        self.mincaps_FP += FP
        self.mincaps_TN += TN
        return mincaps

    def forward(self, batch_ids, GRAPH_SIZES, current_input, last_latent, edge_index, edge_attr, iimask, edge_mask=None):
        SIZE = last_latent.shape[0]
        if self.bits_size is not None:
            current_input = utils.integer2bit(current_input, self.bits_size)
            current_input = self.bit_encoder(current_input)
        else:
            current_input = current_input.unsqueeze(1)

        inp = torch.cat((current_input, last_latent), dim=1)

        encoded_nodes = self.node_encoder(inp)
        if self.steps == 0 and self.bits_size is None: # if we are not using integers we learn infinity embedding for the first step
            encoded_nodes[iimask] = self.infinity
        encoded_edges = self.encode_edges(edge_attr)

        latent_nodes = self.processor(encoded_nodes, encoded_edges, utils.flip_edge_index(edge_index))
        predecessors = self.pred_network(encoded_nodes, latent_nodes, encoded_edges, edge_index)

        output = self.decoder_network(torch.cat((encoded_nodes, latent_nodes), dim=1))
        distances = output.squeeze()
        continue_p = self.get_continue_p(batch_ids, latent_nodes, GRAPH_SIZES)
        return latent_nodes, distances, predecessors, continue_p
