from torch_geometric.data import Data


class FactorGraphDataSet(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None, **kwargs):
        super(FactorGraphDataSet, self).__init__()
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = pos
        for key, item in kwargs.items():
            if key == 'num_nodes':
                self.__num_nodes__ = item
            else:
                self[key] = item

    @property
    def num_nodes(self):
        if hasattr(self, '__num_nodes__'):
            return self.__num_nodes__
        for key, item in self('x', 'batch'):
            return item.size(self.__cat_dim__(key, item))
        return None

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self.__num_nodes__ = num_nodes

    @property
    def num_vars(self):
        if hasattr(self, 'factor_prev_marginals'):
            return self.var_prev_marginal.size(self.__cat_dim__('prv_factor_beliefs', self.var_prev_marginal))
        else:
            return None

    @num_vars.setter
    def num_vars(self, num_vars):
        self.__num_vars__ = num_vars

    @property
    def num_factors(self):
        if hasattr(self, 'factor_prev_marginals'):
            return self.factor_prev_marginals.size(self.__cat_dim__('factor_prev_marginals',
                                                                    self.factor_prev_marginals))
        else:
            return None

    @num_factors.setter
    def num_factors(self, num_factors):
        self.__num_factors__ = num_factors
