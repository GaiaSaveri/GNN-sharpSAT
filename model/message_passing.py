from torch.nn import ReLU

from .mlps import *
from utils.aux_functions import _scatter_log_sum_exp
from .gat import *

damp_param_fixed = 0.5


class BPMessagePassing(torch.nn.Module):
    def __init__(self, neural_bp, n_var_states, transform_factor_var_mlp, transform_var_factor_mlp,
                 damp_param_factor_var, damp_param_var_factor, transform_damp_factor_var, transform_damp_var_factor,
                 attention_factor_var, attention_var_factor, device, debug=True):
        super(BPMessagePassing, self).__init__()

        self.debug = debug
        self.device = device 

        self.transform_factor_var_mlp = transform_factor_var_mlp
        self.transform_var_factor_mlp = transform_var_factor_mlp
        self.attention_factor_var = attention_factor_var
        self.attention_var_factor = attention_var_factor
        if (self.transform_factor_var_mlp and self.attention_factor_var) or \
                (self.transform_var_factor_mlp and self.attention_var_factor):
            raise Exception('Conflicting conditions in transforming messages')

        self.transform_damp_factor_var = transform_damp_factor_var
        self.transform_damp_var_factor = transform_damp_var_factor

        self.damp_param_factor_var = damp_param_factor_var
        self.damp_param_var_factor = damp_param_var_factor

        self.neural_bp = neural_bp
        self.n_var_states = n_var_states
        if neural_bp:
            self.reflected_relu = reflect_xy(ReLU())
            input_dim = n_var_states
            if transform_factor_var_mlp:
                self.mlp_factor_var_msg = MLP(input_dim, [input_dim, input_dim, input_dim],
                                              [self.reflected_relu, self.reflected_relu, self.reflected_relu])

            if attention_factor_var:
                self.gat_factor_var_msg = GAT(num_layers=3, num_heads_layer=[4, 4, 6],
                                              num_features_layer=[input_dim, input_dim, input_dim, input_dim],
                                              bias=True, dropout=0.0, device=self.device)

            if transform_var_factor_mlp:
                self.mlp_var_factor_msg = MLP(input_dim, [input_dim, input_dim, input_dim],
                                              [self.reflected_relu, self.reflected_relu, self.reflected_relu])

            if attention_var_factor:
                self.gat_var_factor_msg = GAT(num_layers=3, num_heads_layer=[4, 4, 6],
                                              num_features_layer=[input_dim, input_dim, input_dim, input_dim],
                                              bias=True, dropout=0.0, device=self.device)

            if transform_damp_factor_var:
                self.mlp_factor_var_damp = MLP(input_dim, [input_dim, input_dim], [])

            if transform_damp_var_factor:
                self.mlp_var_factor_damp = MLP(input_dim, [input_dim, input_dim], [])

    def forward(self, factor_graph, var_factor_prev_msg, factor_var_prev_msg, factor_prev_marginals):
        if self.debug:
            not_nan(factor_prev_marginals)
            not_none(factor_graph)
        # update variable marginals
        factor_var_msg = self.compute_factor_var_msg(factor_prev_marginals=factor_prev_marginals,
                                                     factor_graph=factor_graph,
                                                     var_factor_prev_msg=var_factor_prev_msg,
                                                     factor_var_prev_msg=factor_var_prev_msg)
        factor_var_msg_dim = factor_var_msg.shape

        if self.transform_factor_var_mlp or self.attention_factor_var:
            factor_var_msg = factor_var_msg.view(factor_var_msg_dim[0], factor_graph.n_var_states)
            transformed_factor_var_msg = None
            if self.transform_factor_var_mlp:
                transformed_factor_var_msg = self.mlp_factor_var_msg(factor_var_msg)
            elif self.attention_factor_var:
                msg = torch.cat((factor_var_msg, var_factor_prev_msg))
                transformed_factor_var_msg = self.gat_factor_var_msg((msg, factor_graph, 'factor'))
                n_edges = factor_graph.factor_var_adjacency.shape[1]
                transformed_factor_var_msg = transformed_factor_var_msg.index_select(
                    dim=0, index=torch.arange(0, n_edges).to(self.device))

            if self.debug:
                not_nan(transformed_factor_var_msg)
            factor_var_msg = (1 - damp_param_fixed) * transformed_factor_var_msg + damp_param_fixed * factor_var_msg
            factor_var_msg = clamp_min(factor_var_msg)

            factor_var_msg = factor_var_msg.view(factor_var_msg_dim)

        var_factor_adjacency_arranged = torch.zeros(factor_var_msg.shape, dtype=torch.long)
        for i in range(var_factor_adjacency_arranged.shape[0]):
            var_factor_adjacency_arranged[i] = factor_graph.factor_var_adjacency[1][i]

        var_marginals = torch.zeros(factor_graph.var_prev_marginals.shape, device=self.device).scatter_add(
            dim=0, src=factor_var_msg.to(self.device), index=var_factor_adjacency_arranged.to(self.device))

        if self.debug:
            # var_marginals has shape [# variables, variable cardinality]
            check_dimensionality(var_marginals, factor_graph)
        # normalize variable marginals
        var_marginals = var_marginals - log_sum_exp(var_marginals, dim_to_keep=[0])
        if self.debug:
            check_normalization(var_marginals, -1)
            not_nan(var_marginals)
        # update factor marginals
        var_factor_msg = self.compute_var_factor_msg(var_marginals, factor_graph, factor_var_prev_msg=factor_var_msg,
                                                     var_factor_prev_msg=var_factor_prev_msg)

        if self.transform_var_factor_mlp or self.attention_var_factor:
            var_factor_msg_dim = var_factor_msg.shape
            var_factor_msg = var_factor_msg.view(var_factor_msg_dim[0], factor_graph.n_var_states)
            transformed_var_factor_msg = None
            if self.debug:
                equal_dim(var_factor_msg_dim, tensor_two=factor_var_msg_dim)
            if self.transform_var_factor_mlp:
                transformed_var_factor_msg = self.mlp_var_factor_msg(var_factor_msg)
            elif self.attention_var_factor:
                msg = torch.cat((var_factor_msg, factor_var_prev_msg))
                transformed_var_factor_msg = self.gat_var_factor_msg(data=(msg, factor_graph, 'var'))
                n_edges = factor_graph.factor_var_adjacency.shape[1]
                transformed_var_factor_msg = transformed_var_factor_msg.index_select(
                    dim=0, index=torch.arange(0, n_edges).to(self.device))

            var_factor_msg = (1 - damp_param_fixed) * transformed_var_factor_msg + damp_param_fixed * var_factor_msg

            var_factor_msg = var_factor_msg.view(var_factor_msg_dim)
            var_factor_msg = clamp_min(var_factor_msg)

        # var_factor_msg has shape [# edges, variable cardinality]
        if self.debug:
            check_dimensionality(var_factor_msg, factor_graph)
        # need to make shapes compatible with factor marginals
        var_factor_msg_extra_dims = [factor_graph.n_var_states for i in
                                     range(factor_graph.max_clause_length[0].item() - 1)] + list(var_factor_msg.shape)
        var_factor_msg_full_dim = var_factor_msg.expand(var_factor_msg_extra_dims)
        var_factor_msg_full_dim = var_factor_msg_full_dim.transpose(-2, 0).transpose(-1, 1)
        var_factor_msg_full_dim_flat = var_factor_msg_full_dim.flatten()
        # this is actually just re-arranging and not adding anything
        var_factor_msg_arranged = torch.zeros_like(var_factor_msg_full_dim_flat).scatter_add(
            dim=0, src=var_factor_msg_full_dim_flat, index=factor_graph.indexes_var_factor)

        new_shape = [var_factor_msg.shape[0]] + \
                    [factor_graph.n_var_states for i in range(factor_graph.max_clause_length[0])]
        var_factor_msg_arranged = var_factor_msg_arranged.reshape(new_shape)
        # normalize var marginals
        var_factor_msg_arranged = var_factor_msg_arranged - log_sum_exp(var_factor_msg_arranged, dim_to_keep=[0])

        if self.debug:
            not_nan(var_factor_msg_arranged)

        factor_var_adjacency_arranged = torch.zeros(new_shape, dtype=torch.long)
        for i in range(new_shape[0]):
            factor_var_adjacency_arranged[i] = factor_graph.factor_var_adjacency[0][i]

        factor_marginals = torch.zeros(factor_graph.factor_prev_marginals.shape, device=self.device).scatter_add(
            dim=0, src=var_factor_msg_arranged.to(self.device), index=factor_var_adjacency_arranged.to(self.device))

        if self.debug:
            not_nan(factor_marginals)
        factor_marginals_norm_const = log_sum_exp(factor_marginals, dim_to_keep=[0])
        # dimension for #factors, each state dimension
        if self.debug:
            equal_dim(factor_marginals, factor_graph=factor_graph)
        # normalize factor marginals
        factor_marginals = factor_marginals - factor_marginals_norm_const
        if self.debug:
            not_inf(factor_marginals_norm_const)
            not_nan(factor_marginals)

        set_invalid_positions(factor_marginals, factor_graph)

        return var_factor_msg, factor_var_msg, var_marginals, factor_marginals

    def compute_factor_var_msg(self, factor_prev_marginals, factor_graph, var_factor_prev_msg, factor_var_prev_msg):
        set_invalid_positions(factor_prev_marginals, factor_graph)
        if self.debug:
            check_invalid_positions(factor_prev_marginals, input_graph=factor_graph)

        arranged_factor_marginals = arrange_marginals(factor_prev_marginals, factor_graph, 'factor')
        arranged_factor_valid_configs = arrange_marginals(factor_graph.factor_valid_configs, factor_graph, 'factor')
        if self.debug:
            check_invalid_positions(arranged_factor_marginals, input_mask=arranged_factor_valid_configs)

        set_invalid_positions(arranged_factor_marginals)

        if self.debug:
            check_shapes(arranged_factor_marginals, factor_graph)
            check_valid_values(factor_graph=factor_graph)

        factor_var_msg_bin = _scatter_log_sum_exp(src=arranged_factor_marginals,
                                                  index=factor_graph.factor_var_indices,
                                                  dim_size=factor_graph.factor_var_adjacency.shape[1] *
                                                  factor_graph.n_var_states + 1,
                                                  device=self.device)
        factor_var_msg_no_bin = factor_var_msg_bin[:-1].view(factor_graph.factor_var_adjacency.shape[1],
                                                             factor_graph.n_var_states)

        if self.debug:
            check_valid_values(input_tensor=var_factor_prev_msg)

        # subtract previous messages to avoid double counting the evidence
        factor_var_msg = factor_var_msg_no_bin - var_factor_prev_msg
        factor_var_msg = clamp_min(factor_var_msg)

        if self.transform_damp_factor_var:
            factor_var_msg = factor_var_msg + (1 - self.damp_param_factor_var) * \
                                 (self.mlp_factor_var_damp(factor_var_prev_msg - factor_var_msg) -
                                  self.mlp_factor_var_damp(torch.zeros_like(factor_var_prev_msg)))
        else:
            factor_var_msg = self.damp_param_factor_var * factor_var_msg + \
                                 (1 - self.damp_param_factor_var) * factor_var_prev_msg
        factor_var_msg = clamp_min(factor_var_msg)
        if self.debug:
            not_nan(factor_var_msg)
        return factor_var_msg

    def compute_var_factor_msg(self, var_marginals, factor_graph, factor_var_prev_msg, var_factor_prev_msg):
        mapped_var_marginals = arrange_marginals(var_marginals, factor_graph, 'var')
        if self.debug:
            check_valid_values(input_tensor=factor_var_prev_msg)
        # subtract previous messages to avoid double counting the evidence
        var_factor_msg = mapped_var_marginals - factor_var_prev_msg

        var_factor_msg = clamp_min(var_factor_msg)

        if self.transform_damp_var_factor:
            var_factor_msg = var_factor_msg + (1 - self.damp_param_var_factor) * \
                                 (self.mlp_var_factor_damp(var_factor_prev_msg - var_factor_msg) -
                                  self.mlp_var_factor_damp(torch.zeros_like(var_factor_prev_msg)))
        else:
            var_factor_msg = self.damp_param_var_factor * var_factor_msg + \
                                 (1 - self.damp_param_var_factor) * var_factor_prev_msg
        var_factor_msg = clamp_min(var_factor_msg)
        not_nan(var_factor_msg)
        return var_factor_msg
