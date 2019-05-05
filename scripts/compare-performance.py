import numpy as np
import pandas as pd
import networkx as nx

from main import prepare_data, compute_sigma


def run_high_degree(G, k, l=10000):
    SPREAD = np.zeros(k)

    dict_out_degree = dict(G.out_degree())
    nodes_sorted = sorted(dict_out_degree, key=dict_out_degree.get, reverse=True)
    S = []
    for i in range(k):
        S.append(nodes_sorted[i])

        SPREAD[i] = compute_sigma(G, S, l)

    pd.DataFrame({'k': range(1, k + 1), 'sigma': SPREAD}).to_csv('results/baseline/high_degree.csv', index=False)


def run_central(G, k, l=10000):
    SPREAD = np.zeros(k)

    dict_centrality = nx.closeness_centrality(G)
    nodes_sorted = sorted(dict_centrality, key=dict_centrality.get, reverse=True)
    S = []
    for i in range(k):
        S.append(nodes_sorted[i])

        SPREAD[i] = compute_sigma(G, S, l)

    pd.DataFrame({'k': range(1, k + 1), 'sigma': SPREAD}).to_csv('results/baseline/central.csv', index=False)


def run_random(G, k, l=10000, n=100):
    SPREAD = np.zeros(k)
    for i in range(k):
        mean_spread = 0
        for j in range(n):
            S = list(np.random.choice(G.nodes, size=i + 1, replace=False))
            mean_spread += compute_sigma(G, S, l) / n
        SPREAD[i] = mean_spread

    pd.DataFrame({'k': range(1, k + 1), 'sigma': SPREAD}).to_csv('results/baseline/random.csv', index=False)


if __name__ == "__main__":
    k = 30
    l = 10000

    df_edges, df_nodes, G = prepare_data(
        file_edges='data/df_edges.csv',
        file_nodes='data/df_nodes.csv',
        c=0.3
    )

    run_central(G, k, l)
    # run_high_degree(G, k, l)
    # run_random(G, k, l)
