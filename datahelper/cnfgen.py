import os
import numpy as np
import sys
import argparse

sys.path.append(os.path.abspath('../'))
from utils.file_utils import parse_dimacs
from utils.file_utils import parse_counts
import PyMiniSolvers.minisolvers as minisolvers


def options(parser):
    parser.add_argument('--data_dir', default='../data', type=str, help='Name of the directory to store data')
    parser.add_argument('--count_dir', default='../counts', type=str, help='Name of the directory to store counts')
    parser.add_argument('--id', default=None, type=str, help='Name of the sub-directory for data/counts')
    parser.add_argument('--kind', default='test',  choices=['train', 'test', 'validation'],
                        help='Whether to generate train/test/validation datasets')
    parser.add_argument('--cnfgen_cmd', type=str, help='Command for cnfgen tool')
    parser.add_argument('--k', default=3, type=int, help='k parameter of cnfgen tool')
    parser.add_argument('--n', type=int, help='N parameter of cnfgen tool')
    parser.add_argument('--p', type=float, help='p parameter of cnfgen tool')
    parser.add_argument('--n_problems', default=250, type=int, help='Number of problems to generate')


def rename_files(data_dir, id):
    data_dir = os.path.join(data_dir, id)
    data_files = os.listdir(data_dir)
    for file in data_files:
        if file[-4:] == ".cnf":
            print("renaming")
            name = file[:-4]
            name_dimacs = name + ".dimacs"
            bash = "mv " + data_dir + "/" + file + " " + data_dir + "/" + name_dimacs
            os.system(bash)


def count_generic(data_dir, count_dir, id):
    counts = os.path.join(count_dir, id)
    if not os.path.exists(os.path.join(os.getcwd(), counts)):
        os.makedirs(counts)
    data_dir = os.path.join(data_dir, id)
    data_files = os.listdir(data_dir)
    for file in data_files:
        if file == "parameters.json" or file == ".DS_Store":
            continue
        else:
            count_file = file[:-7]
            path = counts + "/" + count_file + ".txt"
            if os.path.isfile(path):
                continue
            bash = "cd sharpSAT/build/Release; ./sharpsat " + "../../../" + data_dir + "/" + file + " | tail -1 >> " + \
                   "../../../" + path
            os.system(bash)


def cnf_gen(n_problems, data_dir, count_dir, id, cnf_command, type):
    data = os.path.join(data_dir + "/" + id, type)
    if not os.path.exists(os.path.join(os.getcwd(), data_dir + "/" + id + "/" + type)):
        os.makedirs(data_dir + "/" + id + "/" + type)
    counts = os.path.join(count_dir + "/" + id, type)
    if not os.path.exists(os.path.join(os.getcwd(), count_dir + "/" + id + "/" + type)):
        os.makedirs(count_dir + "/" + id + "/" + type)
    sat = True
    i = 0
    while i <= n_problems:
        file = id + "_" + str(i) + ".dimacs"
        path = data + "/" + file
        bash = "cnfgen -q -o" + path + " " + cnf_command
        os.system(bash)
        n_vars, clauses, _ = parse_dimacs(path)
        lengths = [len(clauses[j]) for j in range(len(clauses))]
        lengths = np.array(lengths)
        max_length = np.max(lengths)
        if max_length <= 5:
            solver = minisolvers.MinisatSolver()
            for _ in range(n_vars):
                solver.new_var(dvar=True)
            for c in clauses:
                solver.add_clause(c)
            result = solver.solve()
            if result == sat:
                count_file = file[:-7]
                count_path = counts + "/" + count_file + ".txt"
                if os.path.isfile(count_path):
                    i += 1
                    continue
                bash = "cd ../sharpSAT/build/Release; ./sharpsat " + "../../" + path +\
                       " | tail -1 >> " + "../../" + count_path
                os.system(bash)
                count, _ = parse_counts(counts, count_file + ".txt")
                if count > 1:
                    i += 1


def main(args):
    cnf_gen(args.n_problems, args.data_dir, args.count_dir, args.id, args.cnfgen_cmd + ' ' + str(args.k) + ' gnp '
            + str(args.n) + ' ' + str(args.p), args.kind)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    options(parser)
    main(args=parser.parse_args())
