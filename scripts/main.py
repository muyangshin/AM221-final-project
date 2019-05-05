import argparse
import numpy as np
import pandas as pd
import time
import networkx as nx


def prepare_data(file_edges, file_nodes, c):
    def decide_propagation_probs(df_edges, df_nodes, c=c):
        df = (
            df_edges
            .merge(df_nodes.loc[:, ['node_id', 'ST_WIDTH']].set_index('node_id'), left_on='node_id_to', right_index=True)
            .assign(sum_width=lambda x: x.groupby('node_id_to').ST_WIDTH.transform('sum'))
            .assign(p=lambda x: c * x.ST_WIDTH / x.sum_width)
            .drop(columns=['ST_WIDTH', 'sum_width'])
        )

        return df

    # load data
    df_nodes = (
        pd.read_csv(file_nodes)

        # if ST_WIDTH = 0, use 30 (roughly avg)
        .assign(ST_WIDTH=lambda x: np.where(x.ST_WIDTH == 0, 30, x.ST_WIDTH))
    )
    df_edges = pd.read_csv(file_edges)

    # assign propagation probabilities
    df_edges = decide_propagation_probs(df_edges, df_nodes, c=c)

    # create a graph object
    G = nx.DiGraph(df_edges.loc[:, ['node_id_from', 'node_id_to']].itertuples(index=False))
    for row in df_edges.itertuples():
        G.edges[row.node_id_from, row.node_id_to]['p'] = row.p

    df_edges = df_edges.set_index(['node_id_from', 'node_id_to'])

    return df_edges, df_nodes, G


# Code implementation based on "Improved Algorithms OF CELF and CELF++ for Influence Maximization" (Lv et al., 2014)
# and "Influence Maximization in Python - Greedy vs CELF" (Hautahi Kingi, 2018)
def compute_sigma(G, S, l=10000):
    sigmas = []

    for i in range(0, l):
        active_nodes, new_active_nodes = S[:], S[:]
        while new_active_nodes:
            nodes_just_activated = []
            for node in new_active_nodes:
                neighbors = G[node]
                if len(neighbors) > 0:
                    # do coin flips for its neighbors
                    probs = [neighbors[neighbor]['p'] for neighbor in neighbors]
                    coin_flips = np.random.uniform(0, 1, len(neighbors)) < probs

                    # add activated neighbors to node_just_activated
                    nodes_just_activated += list(np.extract(coin_flips, list(neighbors)))

            # add nodes_just_activated to active_nodes, except those which were already active
            new_active_nodes = list(set(nodes_just_activated) - set(active_nodes))
            active_nodes += new_active_nodes

        sigmas.append(len(active_nodes))

    return np.mean(sigmas)


# Code implementation based on "Improved Algorithms OF CELF and CELF++ for Influence Maximization" (Lv et al., 2014)
# and "Influence Maximization in Python - Greedy vs CELF" (Hautahi Kingi, 2018)
def celf(G, k, l, use_existing=False, k_old=1):
    start_time = time.perf_counter()

    if not use_existing:
        # construct Q for the first iteration
        marginal_gains = [compute_sigma(G, [node], l) for node in G.nodes]
        flags = [0 for node in G.nodes]
        print("one round done: 0")

        Q = sorted(zip(G.nodes, marginal_gains, flags), key=lambda x: x[1], reverse=True)

        S, sigma_previous, sigmas = [Q[0][0]], Q[0][1], [Q[0][1]]
        Q, timelapse = Q[1:], [time.perf_counter() - start_time]

        # save results
        pd.DataFrame({'S': S, 'sigma': sigmas, 'timelapse': timelapse}).to_csv(f'results/results_c_{c}_k_{k}_l_{l}.csv',
                                                                               index=False)
        pd.DataFrame(Q).to_csv(f'results/Q_c_{c}_k_{k}_l_{l}.csv', index=False)
    else:
        # continue the computation from existing Q and results
        df_S = pd.read_csv(f'results/results_c_{c}_k_{k_old}_l_{l}.csv')
        S = df_S['S'].to_list()
        sigmas = df_S['sigma'].to_list()
        sigma_previous = sigmas[-1]
        timelapse = df_S['timelapse'].to_list()

        Q = list(pd.read_csv(f'results/Q_c_{c}_k_{k_old}_l_{l}.csv').itertuples(index=False, name=None))

    i_start = len(S)
    for i in range(i_start, k):
        # find the top element
        while True:
            Q[0] = (
                Q[0][0],
                compute_sigma(G, S + [Q[0][0]], l) - sigma_previous,
                i
            )

            Q = sorted(Q, key=lambda x: x[1], reverse=True)

            if Q[0][2] == i:
                break

        # found the new top element
        sigma_previous += Q[0][1]
        S.append(Q[0][0])
        sigmas.append(sigma_previous)
        timelapse.append(time.perf_counter() - start_time)

        # remove the top element from Q
        Q = Q[1:]

        # save
        pd.DataFrame({'S': S, 'sigma': sigmas, 'timelapse': timelapse}).to_csv(f'results/results_c_{c}_k_{k}_l_{l}.csv', index=False)
        pd.DataFrame(Q).to_csv(f'results/Q_c_{c}_k_{k}_l_{l}.csv', index=False)
        print(f"one round done: {i}")

    print(time.perf_counter() - start_time)

    return S, sigmas, timelapse


if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("c", default=0.3, help="value of c")
    parser.add_argument("k", default=10, help="value of k")
    parser.add_argument("l", default=10000, help="value of l")
    parser.add_argument("use_existing", default=False, help="use existing?")  # TODO: make it optional
    parser.add_argument("k_old", default=10, help="value of k_old")  # TODO: make it optional
    args = parser.parse_args()

    c = float(args.c)
    k = int(args.k)
    l = int(args.l)
    use_existing = (args.use_existing == "True")
    k_old = int(args.k_old)
    print(f"c = {c}, k = {k}, l = {l}")

    # load data
    df_edges, df_nodes, G = prepare_data(
        file_edges='data/df_edges.csv',
        file_nodes='data/df_nodes.csv',
        c=c
    )

    # run CELF
    S_celf, SPREAD_celf, timelapse_celf = celf(G=G, k=k, l=l, use_existing=use_existing, k_old=k_old)
