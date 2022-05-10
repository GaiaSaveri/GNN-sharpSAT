import torch
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
from typing import List
import os
import sys

sys.path.append(os.path.abspath('../'))
from utils.initialization import structure_maps
from utils.file_utils import parse_counts, parse_dimacs


def build_adjacency(formula: List[List[int]]):
    n_clauses = torch.tensor([len(formula)])
    _, adjacency, n_vars = structure_maps(formula)
    return adjacency, n_vars, n_clauses


def formula_from_adjacency(adjacency, clauses_names):
    def clause(a, c): return a[1][torch.where(torch.tensor(a[0].tolist() == c,  dtype=torch.bool))]

    formula = [clause(adjacency, clause_name).tolist() for clause_name in clauses_names]
    return formula


def compute_degrees(adjacency):
    var_names, var_degrees = torch.unique(adjacency[1, :], sorted=True, return_counts=True)
    clauses_names, clauses_degrees = torch.unique(adjacency[0, :], sorted=True, return_counts=True)
    # degree centrality: the degree centrality of a node is the fraction of nodes connected to it
    var_deg_centrality = var_degrees / len(clauses_names)
    clauses_deg_centrality = clauses_degrees / len(var_names)
    return var_names, var_degrees, var_deg_centrality, clauses_names, clauses_degrees, clauses_deg_centrality


def get_sol_time_list(count_dir):
    times = []
    for file in sorted(os.listdir(count_dir)):
        n_solutions, time_exact_solver = parse_counts(count_dir, file)
        times.append(time_exact_solver)
    return times


def two_vars_occurrences(var_names, var_degrees, formula):
    var_names = var_names[torch.where(var_degrees != 1)]
    pairs = torch.combinations(var_names, with_replacement=False)
    occurrences = torch.tensor([[set(p.tolist()) <= set(c) for p in pairs] for c in formula]).sum(dim=0)
    return occurrences


def triangles_helper(pair, c_1, c_2, c_3):
    cond_1 = (pair[0] in c_1) and (pair[1] in c_2)
    cond_2 = (pair[0] in c_2) and (pair[1] in c_1)
    cond_3 = cond_2 and not cond_1
    m = sum([cond_1 and (pair[0] in c_3), cond_1 and (pair[1] in c_3), cond_3 and (pair[0] in c_3),
             cond_3 and (pair[1] in c_3)])
    return m


def triangles(var_names, adjacency):
    triples_vars = torch.combinations(var_names, r=3, with_replacement=False)
    triples_factors = [adjacency[0][torch.where(torch.tensor([{adjacency[1, i].item()} <= set(triple.tolist())
                                                              for i in range(len(adjacency[1]))]))] for triple in
                       triples_vars]
    pairs_factors = [torch.combinations(torch.unique(triple), with_replacement=False) for triple in triples_factors]
    pairs_vars = [torch.combinations(triple, with_replacement=False) for triple in triples_vars]
    n_triangles = 0

    def f(p, i, j):
        return adjacency[0][torch.where(torch.tensor(adjacency[1].tolist() == p[i][j], dtype=torch.bool))]

    def all_checks(p):
        return [f(p, 0, 0), f(p, 0, 1), f(p, 1, 0), f(p, 1, 1), f(p, 2, 0), f(p, 2, 1)]

    def common_f(x, y):
        return torch.cat((x, y)).unique(return_counts=True)[0][
            torch.where(torch.cat((x, y)).unique(return_counts=True)[1] > 1)]

    for i in range(len(pairs_factors)):
        for pair_factor in pairs_factors[i]:
            # take two pairs of variables and check if they go in the pair of factors (distinct)
            # if yes check the third pair
            # if yes add triangle
            z_z, z_o, o_z, o_o, t_z, t_o = all_checks(pairs_vars[i])
            c_z, c_o, c_t = [common_f(z_z, z_o), common_f(o_z, o_o), common_f(t_z, t_o)]
            if c_z.numel() == 0 or c_o.numel() == 0 or c_t.numel() == 0:
                continue
            n_triangles += sum([triangles_helper(pair_factor, c_z, c_o, c_t),
                                triangles_helper(pair_factor, c_o, c_t, c_z),
                                triangles_helper(pair_factor, c_t, c_z, c_o)])
    return n_triangles


def graph_stats(adjacency, var_names, clauses_names):
    g = nx.Graph()
    g.add_nodes_from(clauses_names.tolist(), bipartite=0)
    var = [str(i) for i in var_names]
    g.add_nodes_from(var, bipartite=1)
    g.add_edges_from([(adjacency[0][i].item(), str(adjacency[1][i].item())) for i in range(len(adjacency[0]))])
    # density: ratio of the edges in the graph to the maximum possible number of edges it could have ([0, 1])
    clauses_density = [bipartite.density(g, [i.item()]) for i in clauses_names]
    clauses_density_all = bipartite.density(g, clauses_names.tolist())
    var_density = [bipartite.density(g, i) for i in var]
    var_density_all = bipartite.density(g, var)
    graph_density = nx.density(g)
    n_loops = len(nx.cycle_basis(g))
    return clauses_density, clauses_density_all, var_density, var_density_all, graph_density, n_loops


def produce_header():
    header = ['min_var_degree', 'max_var_degree', 'avg_var_degree', 'std_var_degree', 'min_clause_degree',
              'max_clause_degree', 'avg_clause_degree', 'std_clause_degree', 'min_var_deg_centrality',
              'max_var_deg_centrality', 'avg_var_deg_centrality', 'std_var_deg_centrality',
              'min_clauses_deg_centrality', 'max_clauses_deg_centrality', 'avg_clauses_deg_centrality',
              'std_clauses_deg_centrality', 'min_co_occurrences', 'max_co_occurrences', 'avg_co_occurrences',
              'std_co_occurrences', 'n_triangles', 'min_clauses_density', 'max_clauses_density', 'avg_clauses_density',
              'std_clauses_density', 'all_clauses_density', 'min_var_density', 'max_var_density', 'avg_var_density',
              'std_var_density', 'all_vars_density', 'graph_density', 'n_loops', 'time solver', 'exact_ln',
              'estimated_ln']
    return header


def stats(x):
    return [np.min(x), np.max(x), np.mean(x), np.std(x)]


def produce_row(adjacency):
    var_names, var_degrees, var_deg_centrality, clauses_names, clauses_degrees, clauses_deg_centrality = \
        compute_degrees(adjacency)
    formula = formula_from_adjacency(adjacency, clauses_names)
    co_occurrences = two_vars_occurrences(var_names, var_degrees, formula)
    n_triangles = triangles(var_names, adjacency)
    clauses_density, clauses_density_all, var_density, var_density_all, graph_density, n_loops = \
        graph_stats(adjacency, var_names, clauses_names)

    return stats(var_degrees.numpy()) + stats(clauses_degrees.numpy()) + stats(var_deg_centrality.numpy()) + \
        stats(clauses_deg_centrality.numpy()) + stats(co_occurrences.numpy()) + [n_triangles] + \
        stats(np.array(clauses_density)) + [clauses_density_all] + stats(np.array(var_density)) + \
        [var_density_all, graph_density, n_loops]


def flatten(t):
    return [item for sublist in t for item in sublist]


def summarize_dataset(name, data_dir, count_dir):
    var_deg = []
    clause_deg = []
    var_deg_cent = []
    clauses_deg_cent = []
    n_var = []
    n_clause = []
    two_vars = []
    three_vars = []
    clause_density = []
    vars_density = []
    graphs_density = []
    loops = []
    times = []
    solutions = []
    counts = os.listdir(count_dir)
    for file in os.listdir(data_dir):
        if file[:-7] + ".txt" not in counts:
            # print("skipping")
            continue
        _, formula, _ = parse_dimacs(data_dir + "/" + file)
        adjacency, n_vars, n_clauses = build_adjacency(formula)
        var_names, var_degrees, var_deg_centrality, clauses_names, clauses_degrees, clauses_deg_centrality = \
            compute_degrees(adjacency)
        n_var.append(n_vars.item())
        n_clause.append(n_clauses.item())
        file_count = file[:-7] + ".txt"
        n_sol, time = parse_counts(count_dir, file_count)
        times.append(time)
        solutions.append(n_sol)
        var_deg.append(var_degrees.tolist())
        clause_deg.append(clauses_degrees.tolist())
        var_deg_cent.append(var_deg_centrality.tolist())
        clauses_deg_cent.append(clauses_deg_centrality.tolist())
        formula = formula_from_adjacency(adjacency, clauses_names)
        co_occurrences = two_vars_occurrences(var_names, var_degrees, formula)
        two_vars.append(co_occurrences.tolist())
        _, clauses_density_all, _, var_density_all, graph_density, n_loops = \
            graph_stats(adjacency, var_names, clauses_names)
        clause_density.append(clauses_density_all)
        vars_density.append(var_density_all)
        graphs_density.append(graph_density)
        loops.append(n_loops)
        n_triangles = triangles(var_names, adjacency)
        three_vars.append(n_triangles)
        file_count = file[:-7] + ".txt"
        n_sol, time = parse_counts(count_dir, file_count)
        times.append(time)
        solutions.append(n_sol)
    n_var = np.array(n_var)
    n_clause = np.array(n_clause)
    solutions = np.array(solutions)
    times = np.array(times)
    print(np.mean(n_var), np.mean(n_clause), np.mean(solutions), np.mean(times))
    var_deg = np.array(flatten(var_deg))
    clause_deg = np.array(flatten(clause_deg))
    var_deg_cent = np.array(flatten(var_deg_cent))
    clauses_deg_cent = np.array(flatten(clauses_deg_cent))
    two_vars = np.array(flatten(two_vars))
    return [name] + stats(var_deg) + stats(clause_deg) + stats(var_deg_cent) + stats(clauses_deg_cent) + \
        stats(two_vars) + stats(np.array(clause_density)) + \
        stats(np.array(vars_density)) + stats(np.array(graphs_density)) + stats(np.array(loops)) + \
        stats(np.array(solutions))


def summarize_dataset_header():
    return ['name', 'min_var_degree', 'max_var_degree', 'avg_var_degree', 'std_var_degree', 'min_clause_degree',
            'max_clause_degree', 'avg_clause_degree', 'std_clause_degree', 'min_var_centrality', 'max_var_centrality',
            'avg_var_centrality', 'std_var_centrality', 'min_clause_centrality', 'max_clause_centrality',
            'avg_clause_centrality', 'std_clause_centrality', 'min_co_occurrence', 'max_co_occurrence',
            'avg_co_occurrence', 'std_co_occurrence', 'min_triangles', 'max_triangles', 'avg_triangles',
            'std_triangles', 'min_clause_density', 'max_clause_density', 'avg_clause_density', 'std_clause_density',
            'min_var_density', 'max_var_density', 'avg_var_density', 'std_var_density', 'min_graph_density',
            'max_graph_density', 'avg_graph_density', 'std_graph_density', 'min_loops', 'max_loops', 'avg_loops',
            'std_loops', 'min_solutions', 'max_solutions', 'avg_solutions', 'std_solutions']
