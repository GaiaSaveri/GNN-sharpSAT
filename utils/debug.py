import torch
import numpy as np

ln_zero = -99


def not_nan(input_tensor):
    assert (not torch.isnan(input_tensor).any()), input_tensor


def not_ln_zero(input_tensor):
    assert (not (input_tensor == ln_zero).any()), input_tensor


def not_none(input_tensor):
    assert (input_tensor is not None)


def not_inf(input_tensor):
    assert ((input_tensor != np.inf).all())


def check_normalization(input_tensor, dim):
    test_tensor = torch.sum(torch.exp(input_tensor), dim=dim)
    assert (torch.max(torch.abs(test_tensor - 1)) < .01), \
           (torch.sum(torch.abs(test_tensor - 1)), torch.max(torch.abs(test_tensor - 1)),
            test_tensor)


def check_dimensionality(input_tensor, input_graph):
    assert (len(input_tensor.shape) == 2)
    assert (input_tensor.shape[1] == input_graph.n_var_states), input_tensor.shape


def check_invalid_positions(input_tensor, input_graph=None, input_mask=None):
    if input_mask is None:
        assert ((input_tensor[torch.where(input_graph.factor_valid_configs == 1)] == ln_zero).all())
    else:
        assert ((input_tensor[torch.where(input_mask == 1)] == ln_zero).all())


def check_valid_values(input_tensor=None, factor_graph=None):
    if input_tensor is not None:
        assert ((input_tensor >= ln_zero).all())
    if factor_graph is not None:
        assert ((factor_graph.factor_var_indices <= factor_graph.factor_var_adjacency.shape[1] *
                 factor_graph.n_var_states).all())


def check_shapes(input_tensor, factor_graph):
    assert (input_tensor.view(input_tensor.numel()).shape == factor_graph.factor_var_indices.shape), \
        (input_tensor.view(input_tensor.numel()).shape, factor_graph.factor_var_indices.shape)


def check_equality(tensor_one, tensor_two):
    assert (tensor_one == tensor_two)


def equal_dim(tensor_one, tensor_two=None, factor_graph=None, shape=False):
    if factor_graph is None:
        if shape:
            assert (tensor_one.shape == tensor_two.shape)
        else:
            assert (tensor_one[0] == tensor_two[0]), (tensor_one, tensor_two)
    else:
        assert (len(tensor_one.shape) == (factor_graph.max_clause_length[0] + 1))


def greater_than(input_tensor, value, strict=False):
    if strict:
        assert ((input_tensor > value).all())
    else:
        assert ((input_tensor >= value).all())


def smaller_than(input_tensor, value):
    assert ((input_tensor < value).all())
