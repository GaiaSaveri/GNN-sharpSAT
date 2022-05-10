import os
import random
import sys
import json

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath('../'))
import PyMiniSolvers.minisolvers as minisolvers


class SatProblems:
    def __init__(self, data_dir: str, id: str, n_problems: int, max_cl: int, min_cl: int, max_var: int, min_var: int):
        super(SatProblems, self).__init__()

        if not os.path.exists(os.path.join(os.getcwd(), data_dir)):
            os.makedirs(data_dir)

        self.data_dir = data_dir  # parent directory
        self.id = id  # child directory
        self.n_problems = n_problems  # number of problems generated
        # min - max clause cardinality
        self.min_cl = min_cl
        self.max_cl = max_cl
        # min - max var in each problem
        self.max_var = max_var
        self.min_var = min_var
        # check if data with same parameters already exists
        self.requires_generation = self._check_regeneration()

    @staticmethod
    def _generate_one_clause(n_vars):
        # choose dimension of the clause
        k = min(5, int(1 + np.random.binomial(1, 0.7) + np.random.geometric(0.4)))
        assert n_vars >= k, "n_vars must be greater than k"
        vs = np.random.choice(n_vars, size=k, replace=False)
        return [v + 1 if random.random() < 0.5 else -(v + 1) for v in vs]

    def generate_one_instance(self):
        n_vars = self.set_n_vars()
        n_cl = self.set_n_cl()
        sat = True
        while True:
            clauses = []
            for _ in range(n_cl):
                clauses.append(self._generate_one_clause(n_vars))

            solver = minisolvers.MinisatSolver()
            for _ in range(n_vars):
                solver.new_var(dvar=True)
            for c in clauses:
                solver.add_clause(c)

            result = solver.solve()
            if result == sat:
                return clauses, n_vars, n_cl

    def set_n_vars(self):
        # choose n_vars for this problem, in the range [min_var, max_var]
        return random.randint(self.min_var, self.max_var)

    def set_n_cl(self):
        # choose n_cl for this problem, in the range [min_cl, max_cl]
        return random.randint(self.min_cl, self.max_cl)

    def prepare_data(self, train=False, validation=False, test=False):
        # check if data of same parameter already exists
        regen_train, regen_valid, regen_test = self.requires_generation
        if regen_train and train:
            self._generate_and_save("train")
        if regen_valid and validation:
            self._generate_and_save("validation")
        if regen_test and test:
            self._generate_and_save("test")

        parameters = {
            "id": self.id,
            "n_problems": self.n_problems,
            "min_cl": self.min_cl,
            "max_cl": self.max_cl,
            "min_var": self.min_var,
            "max_var": self.max_var,
        }
        with open(os.path.join(self.data_dir, self.id, "parameters.json"), "w") as f:
            json.dump(parameters, f)

    def _check_regeneration(self):
        # search for existing datasets
        existing_data_dirs = os.listdir(self.data_dir)
        for cur_dir in existing_data_dirs:
            if cur_dir == ".DS_Store":
                continue
            parameters_file = os.path.join(self.data_dir, cur_dir, "parameters.json")
            try:
                with open(parameters_file, "r") as fp:
                    parameters = json.load(fp)
                    n_problems = parameters.get("n_problems")
                    min_cl = parameters.get("min_cl")
                    max_cl = parameters.get("max_cl")
                    max_var = parameters.get("max_var")
                    min_var = parameters.get("min_var")
                    id = parameters.get("id")
                    if id == self.id and n_problems == self.n_problems and min_cl == self.min_cl \
                            and max_cl == self.max_cl and max_var == self.max_var and min_var == self.min_var:
                        # found satisfying dataset
                        self.id = cur_dir
                        modes = os.listdir(os.path.join(self.data_dir, cur_dir))
                        print("Matching Dataset already exists!")
                        return (
                            "train" not in modes,
                            "validation" not in modes,
                            "test" not in modes
                        )  # found corresponding dataset
            except FileNotFoundError:
                continue
        return True, True, True

    def _generate_and_save(self, kind: str):
        data_dir = os.path.join(self.data_dir, self.id, kind)
        os.makedirs(data_dir)

        n_probs = self.n_problems
        if kind == "validation" or kind == "test":
            n_probs = int(n_probs*0.3)

        for i in tqdm(range(n_probs)):
            problem, n_vars, n_cl = self.generate_one_instance()
            out_filename = '{}/cl={:.2f}_var={:.2f}_t={}.dimacs'.format(
                data_dir,
                n_cl,
                n_vars,
                i
            )
            with open(out_filename, 'w') as f:
                f.write("p cnf %d %d\n" % (n_vars, len(problem)))
                for clause in problem:
                    for x in clause:
                        f.write("%d " % x)
                    f.write("0\n")

    def return_path(self):
        return self.data_dir + "/" + self.id
