from torch.utils.data import DataLoader
import os

from .indexing import *
from .initialization import *
from .preprocessing import *
from .file_utils import *
from .base import *


class SatFactorGraph(FactorGraphDataSet):
    def __init__(self, clauses: List[List[int]] = None, ln_sol_count: float = None):
        super(SatFactorGraph, self).__init__()
        if (clauses is not None) and (ln_sol_count is not None):
            self.ln_Z = torch.tensor([ln_sol_count])
            # number of variables in the largest clause
            self.max_factor_dimensions = torch.tensor([5])
            self.max_clause_length = self.max_factor_dimensions  # torch.tensor(SatFactorGraph.get_state_dim(clauses))
            self.n_var_states = torch.tensor([2])  # number of states each variable can take
            # [#factors, 2, ..., 2] (max clause length times)
            self.factor_truth_values, self.factor_valid_configs = preprocess_formula(clauses,
                                                                                     self.max_clause_length.item())
            # [#factors[#var each factor]], [2, #edges], int
            clauses_list, self.factor_var_adjacency, self.n_vars = structure_maps(clauses)
            ordered_var_list = edge_connections(clauses)  # [#edges]
            # [#edges*2^state_dim] index for var to factor messages
            self.indexes_var_factor = indexing_var_factor(ordered_var_list, self.n_var_states.item(),
                                                          self.max_clause_length.item())
            # [#edges*2^state_dim]
            self.factor_var_indices = indexing_factor_var(clauses_list, ordered_var_list, self.n_var_states.item(),
                                                          self.max_clause_length.item())
            # [#factors]
            _, self.factor_degrees = torch.unique(self.factor_var_adjacency[0, :], sorted=True, return_counts=True)
            # [#vars]
            _, self.var_degrees = torch.unique(self.factor_var_adjacency[1, :], sorted=True, return_counts=True)
            self.n_factors = torch.tensor([len(clauses)])  # int
            # [#edges], [#edges], [#vars], [#edges]
            self.var_factor_prev_msg, self.factor_var_prev_msg, self.factor_prev_marginals, \
                self.var_prev_marginals = initialize_messages_marginals(ordered_var_list, self.n_var_states.item(),
                                                                        self.max_clause_length.item(),
                                                                        self.n_factors.item(), self.n_vars.item(),
                                                                        self.factor_valid_configs)
            self.n_edges = torch.tensor([self.factor_var_adjacency.shape[1]])
        else:  # need to do this for batching
            self.ln_Z = None
            self.max_factor_dimensions = None
            self.max_clause_length = None
            self.n_var_states = None
            self.factor_truth_values = None
            self.factor_valid_configs = None
            self.factor_var_adjacency = None
            self.n_vars = None
            self.indexes_var_factor = None
            self.factor_var_indices = None
            self.factor_var_adjacency = None
            self.factor_degrees = None
            self.var_degrees = None
            self.n_factors = None
            self.var_factor_prev_msg = None
            self.factor_var_prev_msg = None
            self.factor_prev_marginals = None
            self.var_prev_marginals = None
            self.n_edges = None

    @staticmethod
    def get_state_dim(clauses: List[List[int]]):
        max_length = -1
        for i in range(len(clauses)):
            if len(clauses[i]) > max_length:
                max_length = len(clauses[i])
        return max_length

    def __inc__(self, key, value):
        if key == 'factor_var_indices':
            return torch.tensor([self.n_var_states * self.factor_var_adjacency.size(1)])
        elif key == 'indexes_var_factor':
            return torch.tensor([self.indexes_var_factor.size(0)])
        elif key == 'factor_var_adjacency':
            return torch.tensor([self.factor_prev_marginals.size(0), self.var_prev_marginals.size(0)]).unsqueeze(dim=1)
        else:
            return super(SatFactorGraph, self).__inc__(key, value)

    def __cat_dim__(self, key, value):
        if key == 'factor_var_adjacency':
            return -1
        else:
            return super(SatFactorGraph, self).__cat_dim__(key, value)


class SatFactorGraphDataSet:
    def __init__(self, data_dir: str, count_dir: str):
        super(SatFactorGraphDataSet, self).__init__()
        # both folders' name already embedded with train/validation/test
        self.data_dir = data_dir
        self.count_dir = count_dir
        self.n_problems = self.set_n_problems()
        self.factor_graphs = self.make_factor_graphs()

    def set_n_problems(self):
        filenames = os.listdir(self.data_dir)
        n_problems = 0
        for file in filenames:
            if file[-6:] == "dimacs":
                n_problems += 1
        return n_problems

    def make_factor_graphs(self):
        problems = []
        filenames = sorted(os.listdir(self.data_dir))
        countnames = sorted(os.listdir(self.count_dir))
        for filename in filenames:
            file = filename[:-7]  # remove .dimacs suffix
            # dimacs and count file should have the same name, in two distinct folders
            count_file = file + ".txt"
            if count_file not in countnames:
                continue
            count, _ = parse_counts(self.count_dir, count_file)
            ln_count = float(np.log(count))
            n_vars, clauses, _ = parse_dimacs(self.data_dir + "/" + filename)
            factor_graph = SatFactorGraph(clauses, ln_count)
            problems.append(factor_graph)
        return problems


class SatFactorGraphsBatch(SatFactorGraph):
    def __init__(self, **kwargs):
        super(SatFactorGraphsBatch, self).__init__(**kwargs)

    @staticmethod
    def from_data_list(data_list, follow_batch=[]):

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = SatFactorGraphsBatch()
        batch.__data_class__ = data_list[0].__class__
        batch.__slices__ = {key: [0] for key in keys}

        for key in keys:
            batch[key] = []

        for key in follow_batch:
            batch['{}_batch'.format(key)] = []

        cumsum = {key: 0 for key in keys}
        batch.batch = []
        # we have a bipartite graph, so keep track of batches for each set of nodes
        batch.batch_factors = []  # for factor beliefs
        batch.batch_vars = []  # for variable beliefs
        junk_bin_val = 0
        for i, data in enumerate(data_list):
            for key in data.keys:
                item = data[key]
                if torch.is_tensor(item) and item.dtype != torch.bool:
                    if key == "factor_var_indices":
                        item = item.clone()  # without this we edit the data for the next epoch, causing errors
                        item[torch.where(item != -1)] = item[torch.where(item != -1)] + cumsum[key]
                    else:
                        item = item + cumsum[key]
                if torch.is_tensor(item):
                    size = item.size(data.__cat_dim__(key, data[key]))
                else:
                    size = 1
                batch.__slices__[key].append(size + batch.__slices__[key][-1])
                if key == "factor_var_indices":
                    factor_var_indices_inc = data.__inc__(key, item)
                    junk_bin_val += factor_var_indices_inc
                    cumsum[key] += factor_var_indices_inc
                else:
                    cumsum[key] += data.__inc__(key, item)
                batch[key].append(item)

                if key in follow_batch:
                    item = torch.full((size,), i, dtype=torch.long)
                    batch['{}_batch'.format(key)].append(item)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes,), i, dtype=torch.long)
                batch.batch.append(item)

            batch.batch_factors.append(torch.full((data.n_factors,), i, dtype=torch.long))
            batch.batch_vars.append(torch.full((data.n_vars,), i, dtype=torch.long))

        if num_nodes is None:
            batch.batch = None

        for key in batch.keys:
            item = batch[key][0]
            if torch.is_tensor(item):
                batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, item))
                if key == "factor_var_indices":
                    batch[key][torch.where(batch[key] == -1)] = junk_bin_val
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])
            else:
                print(key)
                raise ValueError('Unsupported attribute type')

        return batch.contiguous()

    def to_data_list(self):
        if self.__slices__ is None:
            raise RuntimeError('Cannot reconstruct data list from batch')
        keys = [key for key in self.keys if key[-5:] != 'batch']
        cumsum = {key: 0 for key in keys}
        data_list = []
        for i in range(len(self.__slices__[keys[0]]) - 1):
            data = self.__data_class__()
            for key in keys:
                if torch.is_tensor(self[key]):
                    data[key] = self[key].narrow(
                        data.__cat_dim__(key,
                                         self[key]), self.__slices__[key][i],
                        self.__slices__[key][i + 1] - self.__slices__[key][i])
                    if self[key].dtype != torch.bool:
                        data[key] = data[key] - cumsum[key]
                else:
                    data[key] = self[key][self.__slices__[key][i]:self.
                        __slices__[key][i + 1]]
                cumsum[key] = cumsum[key] + data.__inc__(key, data[key])
            data_list.append(data)

        return data_list

    @property
    def num_graphs(self):
        return self.batch[-1].item() + 1


class SatFactorGraphDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, follow_batch=[], **kwargs):
        super(SatFactorGraphDataLoader, self).__init__(dataset, batch_size, shuffle,
                                                       collate_fn=lambda data_list: SatFactorGraphsBatch.from_data_list(
                                                           data_list, follow_batch), **kwargs)


class SatFactorGraphData:
    def __init__(self, data_dir: str, id: str, count_dir: str):
        if not os.path.exists(os.path.join(os.getcwd(), data_dir)):
            os.makedirs(data_dir)
        self.data_dir = data_dir
        self.count_dir = count_dir
        self.id = id

    def get_data_loaders(self, batch_size, train=False, validation=False, test=False):
        if train:  # train loader
            train_data_dir = os.path.join(self.data_dir, self.id, "train")
            train_counts_dir = os.path.join(self.count_dir, self.id, "train")
            train_ds = SatFactorGraphDataSet(data_dir=train_data_dir, count_dir=train_counts_dir)
            train_loader = SatFactorGraphDataLoader(train_ds.factor_graphs, batch_size=batch_size, shuffle=True)
            return train_loader

        if validation:  # validation loader
            validation_data_dir = os.path.join(self.data_dir, self.id, "validation")
            validation_counts_dir = os.path.join(self.count_dir, self.id, "validation")
            validation_ds = SatFactorGraphDataSet(data_dir=validation_data_dir, count_dir=validation_counts_dir)
            validation_loader = SatFactorGraphDataLoader(validation_ds.factor_graphs, batch_size=batch_size,
                                                         shuffle=True)
            return validation_loader

        if test:  # train loader
            test_data_dir = os.path.join(self.data_dir, self.id, "test")
            test_counts_dir = os.path.join(self.count_dir, self.id, "test")
            test_ds = SatFactorGraphDataSet(data_dir=test_data_dir, count_dir=test_counts_dir)
            test_loader = SatFactorGraphDataLoader(test_ds.factor_graphs, batch_size=batch_size, shuffle=True)
            return test_loader
