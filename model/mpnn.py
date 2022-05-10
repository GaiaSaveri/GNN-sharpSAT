from torch.nn import Sequential as Seq, Linear
from torch_geometric.nn import global_add_pool

from .message_passing import BPMessagePassing
from utils import *


class NeuralBP(torch.nn.Module):
    def __init__(self, neural_bp, max_factor_state_dimensions, msg_passing_iters, transform_factor_var_mlp,
                 transform_var_factor_mlp, attention_factor_var, attention_var_factor, learn_bethe, n_var_states,
                 damp_param_factor_var, damp_param_var_factor, transform_damp_factor_var, transform_damp_var_factor,
                 device):
        super().__init__()
        self.msg_passing_iters = msg_passing_iters
        self.learn_bethe = learn_bethe

        self.message_passing_layers = torch.nn.ModuleList([
            BPMessagePassing(neural_bp=neural_bp, n_var_states=n_var_states,
                             transform_factor_var_mlp=transform_factor_var_mlp,
                             transform_var_factor_mlp=transform_var_factor_mlp,
                             transform_damp_factor_var=transform_damp_factor_var,
                             attention_factor_var=attention_factor_var, attention_var_factor=attention_var_factor,
                             transform_damp_var_factor=transform_damp_var_factor,
                             damp_param_factor_var=damp_param_factor_var, damp_param_var_factor=damp_param_var_factor,
                             device=device)
            for i in range(msg_passing_iters)])

        if learn_bethe:
            n_var_states = n_var_states  # 2 for binary variables
            num_ones = (2 * (n_var_states ** max_factor_state_dimensions) + n_var_states)
            mlp_size = msg_passing_iters * num_ones
            self.linear1 = Linear(mlp_size, mlp_size)
            self.linear2 = Linear(mlp_size, 1)
            # if initialize_to_exact_bethe:
            self.linear1.weight = torch.nn.Parameter(torch.eye(mlp_size))
            self.linear1.bias = torch.nn.Parameter(torch.zeros(self.linear1.bias.shape))
            weight_initialization = torch.zeros((1, mlp_size))
            weight_initialization[0, -num_ones:] = 1.0
            self.linear2.weight = torch.nn.Parameter(weight_initialization)
            self.linear2.bias = torch.nn.Parameter(torch.zeros(self.linear2.bias.shape))
            self.final_mlp = Seq(self.linear1, self.linear2)

    def forward(self, factor_graph):
        var_factor_prev_msg = factor_graph.var_factor_prev_msg
        factor_var_prev_msg = factor_graph.factor_var_prev_msg
        factor_prev_marginals = factor_graph.factor_prev_marginals
        var_prev_marginals = factor_graph.var_prev_marginals

        free_energy_components_layers = []

        for message_passing_layer in self.message_passing_layers:

            var_factor_prev_msg, factor_var_prev_msg, var_prev_marginals, factor_prev_marginals = \
                message_passing_layer(factor_graph, var_factor_prev_msg=var_factor_prev_msg,
                                      factor_var_prev_msg=factor_var_prev_msg,
                                      factor_prev_marginals=factor_prev_marginals)
            if self.learn_bethe:
                free_energy_components_cur_layer = self.free_energy_components_concat(
                    factor_marginals=factor_prev_marginals, var_marginals=var_prev_marginals, factor_graph=factor_graph)
                free_energy_components_layers.append(free_energy_components_cur_layer)

        if self.learn_bethe:
            final_mlp_input = torch.cat(free_energy_components_layers, dim=1)
            predicted_ln_z = self.final_mlp(final_mlp_input)
            return predicted_ln_z

        else:
            free_energy_components_last_layer = self.free_energy_components_concat(
                factor_marginals=factor_prev_marginals,
                var_marginals=var_prev_marginals,
                factor_graph=factor_graph)
            predicted_ln_z = torch.sum(free_energy_components_last_layer, dim=1)
            return predicted_ln_z

    @staticmethod
    def average_energy_components(factor_marginals, factor_truth_values, batch_factors, debug=True):
        if debug:
            equal_dim(factor_truth_values, tensor_two=factor_marginals, shape=True)
        avg_energy_components = global_add_pool(torch.exp(factor_marginals) * neg_inf_to_zero(factor_marginals),
                                                batch_factors)
        avg_energy_components = avg_energy_components.view(avg_energy_components.shape[0], -1)
        return avg_energy_components  # negate and sum to get average bethe energy

    @staticmethod
    def entropy_components(factor_marginals, var_marginals, var_degrees, batch_factors, batch_vars, debug=True):
        entropy_components_factors = - global_add_pool(torch.exp(factor_marginals)*neg_inf_to_zero(factor_marginals),
                                                       batch_factors)
        entropy_components_factors = entropy_components_factors.view(entropy_components_factors.shape[0], -1)
        if debug:
            equal_dim(var_marginals.shape, var_degrees.shape)
        entropy_components_vars = global_add_pool(torch.exp(var_marginals) * neg_inf_to_zero(var_marginals) *
                                                  (var_degrees.float() - 1).view(var_degrees.shape[0], 1), batch_vars)
        entropy_components_vars = entropy_components_vars.view(entropy_components_vars.shape[0], -1)
        return entropy_components_factors, entropy_components_vars

    def free_energy_components_concat(self, factor_marginals, var_marginals, factor_graph):
        avg_energy_components = self.average_energy_components(factor_marginals=factor_marginals,
                                                               factor_truth_values=factor_graph.factor_truth_values,
                                                               batch_factors=factor_graph.batch_factors)
        entropy_components_factors, entropy_components_vars = \
            self.entropy_components(factor_marginals=factor_marginals, var_marginals=var_marginals,
                                    var_degrees=factor_graph.var_degrees, batch_factors=factor_graph.batch_factors,
                                    batch_vars=factor_graph.batch_vars)
        if len(avg_energy_components.shape) > 1:
            cat_dim = 1
        else:
            cat_dim = 0
        return torch.cat([avg_energy_components, entropy_components_factors, entropy_components_vars], dim=cat_dim)
