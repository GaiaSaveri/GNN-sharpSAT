from .debug import *


def arrange_marginals(marginals, factor_graph, marginal_type):
    size = [factor_graph.n_factors, factor_graph.n_vars]
    kind_idx = {"factor": 0, "var": 1}
    idx = kind_idx[marginal_type]

    if isinstance(marginals, tuple) or isinstance(marginals, list):
        assert len(marginals) == 2
        if size[1 - idx] != marginals[1 - idx].size(0):
            raise ValueError('Error in tensors dimensions')
        arranged_marginals = marginals[idx]
    else:
        arranged_marginals = marginals.clone()
    not_none(factor_graph)
    arranged_marginals = torch.index_select(arranged_marginals, 0, factor_graph.factor_var_adjacency[idx])
    return arranged_marginals


def max_multiple_dim(input_tensor, axes, keepdim=False):
    output = None
    if keepdim:
        for ax in axes:
            output = input_tensor.max(ax, keepdim=True)[0]
    else:
        for ax in sorted(axes, reverse=True):
            output = input_tensor.max(ax)[0]
    return output


def log_sum_exp(input_tensor, dim_to_keep=None, debug=True):
    if debug:
        not_nan(input_tensor)

    tensor_dimensions = len(input_tensor.shape)
    if debug:
        smaller_than(torch.tensor([dim_to_keep]), tensor_dimensions)
        greater_than(torch.tensor([dim_to_keep]), 0)

    aggregate_dimensions = [i for i in range(tensor_dimensions) if (i not in dim_to_keep)]
    max_values = max_multiple_dim(input_tensor, axes=aggregate_dimensions, keepdim=True)
    max_values[torch.where(max_values == -np.inf)] = 0
    if debug:
        not_nan(max_values)
        greater_than(max_values, -np.inf, strict=True)
        not_nan(input_tensor - max_values)
        not_nan(torch.exp(input_tensor - max_values))
        not_nan(torch.sum(torch.exp(input_tensor - max_values), dim=aggregate_dimensions))
        not_nan(torch.log(torch.sum(torch.exp(input_tensor - max_values), dim=aggregate_dimensions)))

    return_tensor = torch.log(torch.sum(torch.exp(input_tensor - max_values), dim=aggregate_dimensions, keepdim=True)) \
        + max_values
    if debug:
        not_nan(return_tensor)
    return return_tensor


def neg_inf_to_zero(tensor):
    return_tensor = tensor.clone()
    return_tensor[return_tensor == -float('inf')] = 0
    return return_tensor


class reflect_xy(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return -self.func(-x)


def clamp_min(input_tensor):
    output_tensor = input_tensor.clone()
    output_tensor = torch.clamp(output_tensor, min=ln_zero)
    return output_tensor


def set_invalid_positions(tensor_one, input_graph=None, tensor_two=None):
    if input_graph is not None:
        tensor_one[torch.where(input_graph.factor_valid_configs == 1)] = ln_zero
    else:
        if tensor_two is not None:
            tensor_one[torch.where(tensor_two == 1)] = ln_zero
        else:
            tensor_one[torch.where(tensor_one < ln_zero)] = ln_zero


def _scatter_max(src, index, device, dim=-1):
    gather_idx = []
    for i in torch.unique(index, sorted=True):
        gather_idx.append(src[torch.where(index == i)])

    max_idx = []
    for el in gather_idx:
        x = torch.tensor(el.tolist()).to(device)
        if dim == -1:
            max_idx.append(torch.max(x))
        else:
            max_idx.append(torch.max(x.reshape(el.shape), dim=dim)[0])
    if dim == -1:
        maxes = torch.tensor(max_idx).to(device)
        return maxes
    else:
        maxes = torch.cat(max_idx, dim=-1).to(device)
        return maxes


def _scatter_log_sum_exp(src, index, dim_size, device):
    maxes = _scatter_max(src.view(src.numel()), index, device)
    maxes_gathered = maxes.gather(dim=0, index=index)
    recenter_src = src.view(src.numel()) - maxes_gathered
    sums = torch.zeros(dim_size, device=device).scatter_add(src=recenter_src.exp(), index=index, dim=0).to(device)
    out = torch.log(sums + 1e-12) + maxes
    return out


def _scatter_softmax(src, index, device):
    maxes = _scatter_max(src, index, device=device, dim=0)
    maxes = maxes.reshape(torch.unique(index).numel(), src.shape[1])
    index_arranged = torch.zeros_like(src, dtype=torch.long, device=device)
    for i in range(index_arranged.shape[0]):
        index_arranged[i] = index[i]
    maxes_gathered = maxes.gather(dim=0, index=index_arranged)
    recenter_src = src - maxes_gathered
    recenter_src_exp = recenter_src.exp().to(device)
    sum_idx = torch.zeros((torch.unique(index).numel(), recenter_src_exp.shape[1]), dtype=torch.float,
                          device=device).scatter_add(dim=0, src=recenter_src_exp, index=index_arranged)
    den = (sum_idx + 1e-12).gather(dim=0, index=index_arranged)
    out = recenter_src_exp / den
    return out
