import torch
import numpy as np
from typing import List, Tuple

ln_zero = -99


def mark_invalid_configs(config: Tuple[int], valid_config, clause: List[int], max_clause_length: int):
    invalid_config = False
    for dimension in range(len(clause), max_clause_length):  # useless cells
        if config[dimension] == 1:  # a variable in a useless cell is assigned to 1, it might corrupt the result
            invalid_config = True  # this dimension is unused by this clause, set to 0
    if invalid_config:
        valid_config[config] = 1  # the invalid-ness is marked here
    return invalid_config


def assign_clause_sat_config(config: Tuple[int], clause: List[int], clause_sat_config):
    sat = False
    for rel_position, truth_val in enumerate(config):
        # check if at least one assignment satisfies the clause (just one literal is enough)
        if rel_position >= len(clause):  # outside valid locations for the current clauses
            break
        if clause[rel_position] > 0 and truth_val == 1:
            sat = True
        elif clause[rel_position] < 0 and truth_val == 0:
            sat = True
    if sat:
        clause_sat_config[config] = 1
    else:
        clause_sat_config[config] = 0


def preprocess_clause(clause: List[int], max_clause_length: int):
    # Create a tensor for the 2^max_clause_length states
    clause_sat_config = torch.zeros([2 for i in range(max_clause_length)])
    valid_config = torch.zeros([2 for i in range(max_clause_length)])
    # enumerating all possible assignments (for the clause)
    for config in np.ndindex(clause_sat_config.shape):
        invalid_config = mark_invalid_configs(config, valid_config, clause, max_clause_length)
        if invalid_config:
            continue
        assign_clause_sat_config(config, clause, clause_sat_config)
    return clause_sat_config, valid_config


def preprocess_formula(formula: List[List[int]], max_clause_length: int):
    truth_values = []
    valid_configs = []
    for clause in formula:
        clause_sat_config, valid_config = preprocess_clause(clause, max_clause_length)
        truth_values.append(clause_sat_config)
        valid_configs.append(valid_config)
    factor_truth_values = torch.stack(truth_values, dim=0)
    factor_valid_configs = torch.stack(valid_configs, dim=0)
    log_truth_values = torch.log(factor_truth_values)
    log_truth_values[torch.where(log_truth_values == -np.inf)] = ln_zero
    expansion_list = list(factor_truth_values.shape)
    log_truth_values = log_truth_values.expand(expansion_list)
    factor_valid_configs = factor_valid_configs.expand(expansion_list)
    return log_truth_values, factor_valid_configs
