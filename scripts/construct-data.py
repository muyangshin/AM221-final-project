import numpy as np
import pandas as pd

# construct nodes
df_nodes = (
    pd.read_csv("data/Centerline.csv")
    .query('BOROCODE == 1')  # Manhattan
    .query('STATUS == 2')  # constructed streets
    .query('TRAFDIR != "NV"')  # exclude non-vehicular

    # ST_WIDTH half if TW
    .assign(ST_WIDTH=lambda x: np.where(x.TRAFDIR == "TW", x.ST_WIDTH / 2, x.ST_WIDTH))
)

df_nodes['geom_list'] = df_nodes.the_geom.str.replace('MULTILINESTRING ((', '', regex=False).str.replace('))', '', regex=False).str.split(',')
df_nodes['geom_0'] = df_nodes.geom_list.map(lambda x: x[0]).str.strip()
df_nodes['geom_1'] = df_nodes.geom_list.map(lambda x: x[-1]).str.strip()

# transform to one-way segments
df_nodes.geom_0, df_nodes.geom_1 = np.where(df_nodes.TRAFDIR == 'TF', [df_nodes.geom_1, df_nodes.geom_0], [df_nodes.geom_0, df_nodes.geom_1])
df_nodes = df_nodes.append(df_nodes.query('TRAFDIR == "TW"').assign(geom_0=df_nodes.geom_1, geom_1=df_nodes.geom_0), sort=False, ignore_index=True)

df_nodes = df_nodes.loc[:, ['PHYSICALID', 'ST_LABEL', 'geom_0', 'geom_1', 'ST_WIDTH', 'SHAPE_Leng']].rename_axis('node_id').reset_index()
df_nodes = df_nodes.drop_duplicates(subset=['geom_0', 'geom_1'])

# construct edges
df_edges = (
    df_nodes
    .merge(df_nodes, how='inner', left_on='geom_1', right_on='geom_0', suffixes=('_from', '_to'))

    # assume no U-turn
    .query('PHYSICALID_from != PHYSICALID_to')

    .loc[:, ['node_id_from', 'node_id_to']]
)

df_edges_to_remove = (
    df_edges
    .assign(
        count_from=lambda x: x.groupby('node_id_from').node_id_to.transform('count'),
        count_to=lambda x: x.groupby('node_id_to').node_id_from.transform('count')
    )
    .query('count_from == 1 and count_to == 1')
    .set_index('node_id_from')
    .loc[:, ['node_id_to']]
)

# if an edge is the only edge both for from- and to-node, collapse two nodes into one
# TODO: deal with multiple values of width and length
df_edges_cleaned = df_edges
while True:
    df_edges_cleaned = (
        df_edges_cleaned
        .merge(df_edges_to_remove.rename(columns={'node_id_to': 'node_id_from_cleaned'}), how='left', left_on='node_id_from', right_index=True)
        .merge(df_edges_to_remove.rename(columns={'node_id_to': 'node_id_to_cleaned'}), how='left', left_on='node_id_to', right_index=True)
    )

    if (~pd.isna(df_edges_cleaned.node_id_from_cleaned)).sum() + (~pd.isna(df_edges_cleaned.node_id_to_cleaned)).sum() == 0:
        df_edges_cleaned = df_edges_cleaned.drop(columns=['node_id_from_cleaned', 'node_id_to_cleaned'])
        break

    df_edges_cleaned = (
        df_edges_cleaned
        .assign(
            node_id_from=lambda x: np.where(pd.isna(x.node_id_from_cleaned), x.node_id_from, x.node_id_from_cleaned),
            node_id_to=lambda x: np.where(pd.isna(x.node_id_to_cleaned), x.node_id_to, x.node_id_to_cleaned)
        )
        .drop(columns=['node_id_from_cleaned', 'node_id_to_cleaned'])
        .drop_duplicates()
    )

df_edges = df_edges_cleaned.query('node_id_from != node_id_to')
df_nodes = df_nodes.loc[lambda x: np.logical_or(x.node_id.isin(df_edges.node_id_from), x.node_id.isin(df_edges.node_id_to))]

# clean index: consecutive from 0
df_nodes = (
    df_nodes
    .reset_index().reset_index()
    .rename(columns={'level_0': 'node_id_0'})
    .drop(columns=['index'])
)

df_edges = (
    df_edges
    .merge(df_nodes.loc[:, ['node_id', 'node_id_0']].set_index('node_id'), left_on='node_id_from', right_index=True)
    .drop(columns=['node_id_from'])
    .rename(columns={'node_id_0': 'node_id_from'})
    .merge(df_nodes.loc[:, ['node_id', 'node_id_0']].set_index('node_id'), left_on='node_id_to', right_index=True)
    .drop(columns=['node_id_to'])
    .rename(columns={'node_id_0': 'node_id_to'})
)

df_nodes = df_nodes.drop(columns=['node_id']).rename(columns={'node_id_0': 'node_id'})

# reverse df_edges: congestion propagation works the other way round
df_edges = df_edges.rename(columns={
    'node_id_from': 'node_id_to',
    'node_id_to': 'node_id_from'
})

# save
df_nodes.to_csv('data/df_nodes.csv', index=False)
df_edges.to_csv('data/df_edges.csv', index=False)
