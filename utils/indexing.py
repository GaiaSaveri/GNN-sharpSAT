from typing import List
from .debug import *


def indexing_var_factor_helper(var_correct_idx: int, n_var_states: int, max_clause_length: int, offset: int):
    # [offset, offset+1, ..., offset+2^state_dim-1]
    l = torch.tensor([i + offset for i in range(n_var_states ** max_clause_length)])
    l_shape = [n_var_states for i in range(max_clause_length)]
    l = l.reshape(l_shape)
    l = l.transpose(var_correct_idx, 0)
    return l.flatten()


def indexing_var_factor(ordered_var_list: List[int], n_var_states: int, max_clause_length: int):
    offset_idx_var_factor = []
    for pos_idx, var_idx in enumerate(ordered_var_list):
        # each var has 2^state_dim slots
        var_offset = pos_idx * (n_var_states ** max_clause_length)
        var_indices = indexing_var_factor_helper(var_correct_idx=var_idx, n_var_states=n_var_states,
                                                 max_clause_length=max_clause_length, offset=var_offset)
        var_indices_cloned = var_indices.clone()
        offset_idx_var_factor.append(var_indices_cloned)
    indexes_var_factor = torch.cat(offset_idx_var_factor)
    return indexes_var_factor


def indexing_factor_var(clauses_list: List[List[int]], ordered_var_list: List[int], n_var_states: int,
                        max_clause_length: int, debug=True):
    n_messages = 0
    for clause_vars in clauses_list:
        n_messages += len(clause_vars)  # total number of messages
    factor_var_indices_temp = []
    junk_bin = n_var_states * len(ordered_var_list)
    pos_idx = torch.arange(n_var_states ** max_clause_length)
    msg_idx = 0
    for clause, clause_vars in enumerate(clauses_list):
        n_unused_vars = max_clause_length - len(clause_vars)
        for relative_var_idx, var_idx in enumerate(clause_vars):
            factor_var_indices_clause = ln_zero * torch.ones(n_var_states ** max_clause_length, dtype=torch.long)
            # depending on the relative position of ech var inside a clause, its assignment changes with different speed
            rel_left_variation_speed = n_var_states ** (max_clause_length - relative_var_idx - 1)
            for var_state_idx in range(n_var_states):
                factor_var_indices_clause[
                    ((pos_idx // rel_left_variation_speed) % n_var_states) == var_state_idx] = msg_idx + var_state_idx
            if debug:
                not_ln_zero(factor_var_indices_clause)
            # send unused factor states to the junk bin
            if n_unused_vars > 0:
                # when in presence of a clause with less variables than state_dimension
                # we need to consider that the assignment are duplicated
                # it doubles at each iteration
                rel_right_variation_speed = 1
                for unused_var_idx in range(n_unused_vars):
                    # send all factor states corresponding to the unused variable being in any state except 0 to
                    # the junk bin
                    for var_state in range(1, n_var_states):
                        factor_var_indices_clause[
                            ((pos_idx // rel_right_variation_speed) % n_var_states) == var_state] = junk_bin
                    rel_right_variation_speed *= n_var_states
            factor_var_indices_clause_cloned = factor_var_indices_clause.clone()
            factor_var_indices_temp.append(factor_var_indices_clause_cloned)
            msg_idx += n_var_states
            factor_var_indices_clause[torch.where(factor_var_indices_clause != -1)] += n_var_states
    if debug:
        check_equality(msg_idx, n_var_states * n_messages)
        check_equality(len(factor_var_indices_temp), n_messages)

    factor_var_indices = torch.cat(factor_var_indices_temp)
    return factor_var_indices
