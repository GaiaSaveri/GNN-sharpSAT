import numpy as np
import os
import sys

sys.path.append(os.path.abspath('../'))
from utils.file_utils import parse_counts


def parse_losses(name, id):
    folder = "../losses/" + name
    with open(folder + "/" + id + "_testing_results.txt", "r") as f:
        for line in f:
            l = line.strip().split(" ")
            loss = l[0]
        return float(loss)


def parse_results(name, id):
    exact_ln = []
    estimated_ln = []
    lines = []
    folder = "../results/" + name
    with open(folder + "/" + id + ".txt", "r") as f:
        l = f.readlines()
        for ll in l:
            line = ll.strip().split()
            lines.append(line)
    for line in lines:
        exact_ln.append(float(line[0]))
        estimated_ln.append(float(line[1]))
    exact_ln = np.array(exact_ln)
    estimated_ln = np.array(estimated_ln)
    diff = estimated_ln - exact_ln
    return exact_ln, estimated_ln, diff


def relative_error(exact, estimated):
    MRE = []
    for i in range(len(exact)):
        diff = np.abs(exact[i] - (estimated[i]))
        err = diff / exact[i]
        MRE.append(err)
    MRE = np.array(MRE)
    return MRE


def mre(name, id):
    exact, estimated, _ = parse_results(name, id)
    err = relative_error(exact, estimated)
    mean_err = np.mean(err)
    return mean_err


def rmse(log_exact, log_bp):
    squares = []
    for i in range(len(log_bp)):
        squares.append((log_exact[i]-log_bp[i])**2)
    squares = np.array(squares)
    rmse = np.mean(squares)
    return rmse


def parse_approx(name):
    count_folder = "../counts/" + name + "/test"
    approx_dir = "../approx/" + name
    files = os.listdir(approx_dir)
    approx = []
    counts = []
    for file in files:
        if file == ".DS_Store":
            continue
        if not os.path.exists(count_folder + "/" + file):
            continue
        with open(approx_dir + "/" + file, 'r') as f:
            for line in f:
                l = line.strip().split(" ")
                if l[0] != "With":
                    continue
                assert (l[0] == "With"), l
                assert (l[1] == "confidence"), l
                if l[2] != "at":
                    continue
                assert (l[3] == "least:0.8"), l
                assert (l[4]== ""), l
                assert (l[5] == "Approximate"), l
                assert (l[6] == "count"), l
                assert (l[7] == "is"), l
                i = 0
                while l[8][i] != "x":
                    i += 1
                fact = int(l[8][1:i])
                assert(int(l[8][i+1]) == 2), l
                exp = int(l[8][i+3:])
                approx.append(fact*(2**exp))
            count, _ = parse_counts(count_folder, file)
            counts.append(count)
    approx = np.array(approx)
    counts = np.array(counts)
    mse = rmse(np.log(counts), np.log(approx))
    mre = relative_error(np.log(counts), np.log(approx))
    rel_error = np.mean(mre)
    return [mse, rel_error]
