import torch
from typing import List


class MLP(torch.nn.Module):
    def __init__(self, d_in: int, d_outs: List[int], act: List, exact=True):
        super(MLP, self).__init__()
        self.d_in = d_in
        self.d_outs = d_outs

        self.linears = torch.nn.ModuleList()
        self.activation_func = act

        self._initialize_layers(exact)

    def _initialize_layers(self, exact):
        d_in = self.d_in
        for d_out in self.d_outs:
            l = torch.nn.Linear(d_in, d_out)
            # initialize exact BP
            if exact:
                l.weight = torch.nn.Parameter(torch.eye(d_out))
                l.bias = torch.nn.Parameter(torch.zeros(l.bias.shape))
            self.linears.append(l)
            d_in = d_out

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if i < len(self.linears) and self.activation_func:
                x = self.activation_func[i](x)
        return x
