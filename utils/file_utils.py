from collections import defaultdict


def parse_counts(count_dir, file):
    with open(count_dir + "/" + file, 'r') as f:
        for line in f:
            l = line.strip().split(" ")
            assert (l[0] == "#"), l
            assert (l[1] == "solutions"), l
            count = float(l[2])
            assert (l[3] == "time:"), l
            time = float(l[4])
    return count, time


def parse_dimacs(filename):
    # parse a single dimacs file
    clauses = []
    vars_dict = defaultdict(int)  # how many times each variable appears
    with open(filename, 'r') as f:
        for line in f:
            l = line.strip().split()
            if len(l) == 0:
                continue
            if l[0] == "p":  # problem line
                n_vars = int(l[2])
            elif l[0] == "c":  # comment line
                continue
            else:
                clause = [int(s) for s in l[:-1]]
                for var in clause:
                    vars_dict[int(abs(var))] += 1
                clauses.append(clause)
    for var_name, var_degree in vars_dict.items():
        assert (n_vars >= var_name >= 1)
    for var_name in range(1, n_vars + 1):
        if var_name not in vars_dict:
            clauses.append([-var_name, var_name])
            vars_dict[var_name] = 2
    loaded = False
    if len(vars_dict) == n_vars:
        loaded = True
    return n_vars, clauses, loaded
