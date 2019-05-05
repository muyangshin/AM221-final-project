import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from scipy.stats import spearmanr


# load data
df_nodes = (
    pd.read_csv('data/df_nodes.csv')

    # if ST_WIDTH = 0, use 30 (roughly avg)
    .assign(ST_WIDTH=lambda x: np.where(x.ST_WIDTH == 0, 30, x.ST_WIDTH))
)
df_edges = pd.read_csv('data/df_edges.csv')

G = nx.DiGraph(df_edges.loc[:, ['node_id_from', 'node_id_to']].itertuples(index=False))

df_c01 = pd.read_csv('results/results_c_0.1_k_100_l_10000.csv')
df_c03 = pd.read_csv('results/results_c_0.3_k_100_l_10000.csv')
df_c05 = pd.read_csv('results/results_c_0.5_k_100_l_10000.csv')

# Top 20 nodes compared with different values of c
df_table = (
    df_c03.loc[:, ['S']].assign(rank=lambda x: x.index + 1).iloc[:20]
    .merge(df_c01.loc[:, ['S']].assign(rank_c01=lambda x: (x.index + 1).astype(str)),
           how='left', left_on='S', right_on='S')
    .merge(df_c05.loc[:, ['S']].assign(rank_c05=lambda x: (x.index + 1).astype(str)),
           how='left', left_on='S', right_on='S')
    .assign(rank_c01=lambda x: np.where(pd.isna(x.rank_c01), '', x.rank_c01))
    .assign(rank_c05=lambda x: np.where(pd.isna(x.rank_c05), '', x.rank_c05))
)
df_table.columns = ['Node id', 'Rank (c=0.3)', 'Rank (c=0.1)', 'Rank (c=0.5)']

df_table.to_csv('results/top_20_nodes_with_different_c.csv', index=False)

# VS BASELINE MODELS
df_central = pd.read_csv('results/baseline/central.csv')
df_high_degree = pd.read_csv('results/baseline/high_degree.csv')
df_random = pd.read_csv('results/baseline/random.csv')

df_spread = (
    pd.DataFrame({
        'Greedy': df_c03.sigma,
        'High Degree': df_high_degree.sigma,
        'Central': df_central.sigma,
        'Random': df_random.sigma,
    })
    .assign(k=lambda x: x.index + 1)
    .iloc[:30]
)
plt.figure(figsize=(15, 12))
lines = df_spread.plot.line(x='k')
plt.xlabel('Seed Size (k)')
plt.ylabel('Spread (sigma)')

plt.savefig('plots/vs_baseline_models.png')
