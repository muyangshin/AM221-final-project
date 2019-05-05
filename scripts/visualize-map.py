import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely import wkt


def get_gdf_nodes(filename):
    df_nodes = (
        pd.read_csv('data/df_nodes.csv')
            .merge(pd.read_csv('data/Centerline.csv').loc[:, ['PHYSICALID', 'the_geom']], how='left', on='PHYSICALID')
            .merge(pd.read_csv(filename).assign(round=lambda x: x.index + 1).loc[:,
                   ['S', 'round']],
                   how='left', left_on='node_id', right_on='S')
            .assign(round_group=lambda x: np.where(
            pd.isna(x['round']),
            0,
            np.where(
                x['round'] <= 30,
                3,
                np.where(
                    x['round'] <= 50,
                    2,
                    1
                )
            )
        ))
            .query('PHYSICALID != 92825')
    )

    df_nodes['the_geom_clean'] = df_nodes.the_geom.apply(wkt.loads)
    # df_nodes2 = df_nodes.query('round > 0')

    return gpd.GeoDataFrame(df_nodes, geometry='the_geom_clean')


gdf_nodes_c01 = get_gdf_nodes('results/results_c_0.1_k_100_l_10000.csv')
gdf_nodes_c03 = get_gdf_nodes('results/results_c_0.3_k_100_l_10000.csv')
gdf_nodes_c05 = get_gdf_nodes('results/results_c_0.5_k_100_l_10000.csv')

f, axarr = plt.subplots(3, 1)
f.set_figwidth(4)
f.set_figheight(10)

gdf_nodes_c03.plot(ax=axarr[0], column='round_group', cmap='Reds', markersize=10)
axarr[0].set_title("c = 0.3")
axarr[0].axis('off')

gdf_nodes_c01.plot(ax=axarr[1], column='round_group', cmap='Reds', markersize=10)
axarr[1].set_title("c = 0.1")
axarr[1].axis('off')

gdf_nodes_c05.plot(ax=axarr[2], column='round_group', cmap='Reds', markersize=10)
axarr[2].set_title("c = 0.5")
axarr[2].axis('off')

plt.savefig('paper/plots/maps.png', bbox_inches='tight', dpi=300)
