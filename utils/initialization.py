from typing import List
from collections import defaultdict
from .aux_functions import *


def structure_maps(formula: List[List[int]]):
    # list of [factor_idx, var_idx] for each edge factor to variable edge
    # factor_idx is given by the order in which clauses appear in the formula
    factor_var_adjacency_temp = []
    vars_occurrences = defaultdict(int)  # keeping the count of how many times each variable appears
    # clauses_list[i] is a list of all variables that factor with index i shares an edge with
    # clauses_list[i][j] is the index of the jth variable that the factor with index i shares an edge with
    clauses_list = []
    for clause_idx, clause in enumerate(formula):
        vars_in_clause = []  # idx of the variables appearing in the current clause
        for literal in clause:
            var_idx = np.abs(literal) - 1  # start enumerating from 0 (in dimacs from 1)
            factor_var_adjacency_temp.append([clause_idx, var_idx])
            vars_occurrences[np.abs(literal)] += 1  # one more occurrence of that variable
            vars_in_clause.append(var_idx)
        clauses_list.append(vars_in_clause)
    factor_var_adjacency = torch.tensor(factor_var_adjacency_temp, dtype=torch.long)
    factor_var_adjacency = factor_var_adjacency.t().contiguous()
    n_vars = len(vars_occurrences)
    n_vars = torch.tensor([n_vars])
    return clauses_list, factor_var_adjacency, n_vars


def edge_connections(clauses, max_clause_degree=None):
    ordered_var_list = []
    for clause in clauses:
        for var_idx in range(len(clause)):
            ordered_var_list.append(var_idx)  # source node is the factor
            if max_clause_degree is not None:
                assert (var_idx < max_clause_degree), (var_idx, max_clause_degree)
    return ordered_var_list


def initialize_messages_marginals(edge_var_indices: List[int], n_var_states: int, max_clause_length: int,
                                  n_factors: int, n_vars: int, factor_valid_configs, initialize_randomly=False,
                                  debug=True):
    n_edges = len(edge_var_indices)
    var_factor_prev_msg = torch.log(torch.stack(
        [torch.ones([n_var_states], dtype=torch.float) for j in range(n_edges)],
        dim=0))
    factor_var_prev_msg = torch.log(torch.stack(
        [torch.ones([n_var_states], dtype=torch.float) for j in range(n_edges)],
        dim=0))
    factor_prev_marginals = torch.log(torch.stack([torch.ones(
        [n_var_states for i in range(max_clause_length)], dtype=torch.float) for j in range(n_factors)], dim=0))
    var_prev_marginals = torch.log(torch.stack(
        [torch.ones([n_var_states], dtype=torch.float) for j in range(n_vars)], dim=0))
    if initialize_randomly:
        var_factor_prev_msg = torch.rand_like(var_factor_prev_msg)
        factor_var_prev_msg = torch.rand_like(factor_var_prev_msg)
        factor_prev_marginals = torch.rand_like(factor_prev_marginals)
        var_prev_marginals = torch.rand_like(var_prev_marginals)

    if debug:
        equal_dim(factor_prev_marginals, tensor_two=factor_valid_configs, shape=True)
    # unused locations
    set_invalid_positions(factor_prev_marginals, tensor_two=factor_valid_configs)
    return var_factor_prev_msg, factor_var_prev_msg, factor_prev_marginals, var_prev_marginals
