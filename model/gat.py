import torch.nn as nn

from utils.aux_functions import _scatter_softmax
from utils import *


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, activation, concat, dropout, bias, device):
        # in_features: number of input features
        # out_features: number of output features
        # num_heads: number of attention heads
        # concat: whether to concatenate or average
        super(GATLayer, self).__init__()
        self.device = device
        self.concat = concat
        self.num_heads = num_heads
        self.out_features = out_features
        # shared linear transformation, parametrized by W matrix (actually num_heads independent matrices)
        self.W = nn.Linear(in_features, num_heads * out_features, bias=False)
        # divide trainable weight a into left (target) and right (source) part (instead of concatenation)
        self.a_target = nn.Parameter(torch.Tensor(1, num_heads, out_features))
        self.a_source = nn.Parameter(torch.Tensor(1, num_heads, out_features))
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_heads * out_features))
            nn.init.zeros_(self.bias)
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        self.leakyReLU = nn.LeakyReLU(0.2)  # as in the GAT paper
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_target)
        nn.init.xavier_uniform_(self.a_source)

    def forward(self, data):
        messages, factor_graph, msg_type = data
        # msg_type: 'factor' --> factor is sending message, 'var' --> variable is sending message
        mapping_indices = {"factor": 0, "var": 1}
        idx = mapping_indices[msg_type]
        index = factor_graph.factor_var_adjacency[idx]
        n_edges = factor_graph.factor_var_adjacency.shape[1]
        select_source_idx = torch.arange(0, n_edges).to(self.device)
        select_target_idx = torch.arange(n_edges, 2*n_edges).to(self.device)
        # messages shape is [#edges, cardinality] for both kind of messages
        messages_source = self.dropout(messages.index_select(dim=0, index=select_source_idx))  # [#edges, in_dim]
        messages_target = self.dropout(messages.index_select(dim=0, index=select_target_idx))
        # projection --> shape [#edges, num_heads, out_dim]
        messages_proj_source = self.W(messages_source).view(-1, self.num_heads, self.out_features)
        messages_proj_source = self.dropout(messages_proj_source)
        messages_proj_target = self.W(messages_target).view(-1, self.num_heads, self.out_features)
        messages_proj_target = self.dropout(messages_proj_target)
        # edge attention calculation  --> shape [edges, num_heads]
        # messages to be transformed
        score_source = (messages_proj_source * self.a_source).sum(dim=-1)
        # messages to be weighted
        score_target = (messages_proj_target * self.a_target).sum(dim=-1)
        score_edges = self.leakyReLU(score_source + score_target)
        assert (score_edges.shape[0] == n_edges)
        # neighborhood-wise softmax --> shape [edges, num_heads, 1]
        attentions = _scatter_softmax(src=score_edges, index=index, device=self.device)
        attentions = attentions.unsqueeze(-1)
        assert (attentions.shape[0] == n_edges)
        # aggregation --> shape [edges, num_heads, out_dim]
        weighted_messages_sum = attentions * messages_proj_target
        if self.concat:
            # shape [edges, num_heads * out_dim]
            weighted_messages_sum = weighted_messages_sum.view(-1, self.num_heads * self.out_features)
            messages_target_placeholder = messages_proj_target.view(-1, self.num_heads * self.out_features)
        else:
            # if not concat, need to average across all heads
            # shape [edges, out_dim]
            weighted_messages_sum = weighted_messages_sum.mean(dim=1)
            messages_target_placeholder = messages_proj_target.mean(dim=1)
        if self.activation is not None:
            weighted_messages_sum = self.activation(weighted_messages_sum)
        return torch.cat((weighted_messages_sum, messages_target_placeholder)), factor_graph, msg_type


class GAT(nn.Module):
    def __init__(self, num_layers, num_heads_layer, num_features_layer, bias, dropout, device):
        super(GAT, self).__init__()
        assert num_layers == len(num_heads_layer) == len(num_features_layer) - 1
        num_heads_layer = [1] + num_heads_layer
        gat_layers = []  # collect gat layers
        for i in range(num_layers):
            layer = GATLayer(
                in_features=num_features_layer[i] * num_heads_layer[i],
                out_features=num_features_layer[i+1],
                num_heads=num_heads_layer[i+1],
                activation=nn.ELU() if i < num_layers - 1 else None,
                concat=True if i < num_layers - 1 else False,
                dropout=dropout,
                bias=bias,
                device=device
            )
            gat_layers.append(layer)
        self.gat_net = nn.Sequential(*gat_layers)

    def forward(self, data):
        data = self.gat_net(data)
        messages, _, _ = data
        return messages
