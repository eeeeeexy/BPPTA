""" Build the user-agnostic global trajectory flow map from the sequence data """
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


def build_global_task_checkin_graph(df, exclude_user=None):
    G = nx.DiGraph()
    users = list(set(df['user_id'].to_list()))

    if exclude_user in users: users.remove(exclude_user)
    loop = tqdm(users)
    for user_id in loop:
        user_df = df[df['user_id'] == user_id]
        # Add node (POI)
        for i, row in user_df.iterrows():
            node = row['poi_id']
            if node not in G.nodes():
                G.add_node(row['poi_id'],
                           checkin_cnt=1,
                           poi_catid=row['poi_catid'],
                           latitude=row['latitude'],
                           longitude=row['longitude'])
            else:
                G.nodes[node]['checkin_cnt'] += 1

        # Add edges (Check-in seq)
        previous_poi_id = 0
        # previous_traj_id = 0
        for i, row in user_df.iterrows():
            poi_id = row['poi_id']
            # traj_id = row['trajectory_id']
            # No edge for the begin of the seq or different traj
            if (previous_poi_id == 0):
            # if (previous_poi_id == 0) or (previous_traj_id != traj_id):
                previous_poi_id = poi_id
                # previous_traj_id = traj_id
                continue

            # Add edges
            if G.has_edge(previous_poi_id, poi_id):
                G.edges[previous_poi_id, poi_id]['weight'] += 1
            else:  # Add new edge
                G.add_edge(previous_poi_id, poi_id, weight=1)
            # previous_traj_id = traj_id
            previous_poi_id = poi_id

    print(np.array(G), np.array(G).shape)

    return G


def save_task_graph_to_csv(G, dst_dir):
    # Save graph to an adj matrix file and a nodes file
    # Adj matrix file: edge from row_idx to col_idx with weight; Rows and columns are ordered according to nodes file.
    # Nodes file: node_name/poi_id, node features (category, location); Same node order with adj matrix.

    # Save adj matrix
    nodelist = G.nodes()
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    print(np.array(A))
    # np.save(os.path.join(dst_dir, 'adj_mtx.npy'), A.todense())
    np.savetxt(os.path.join(dst_dir, 'task_graph_A.csv'), A.todense(), delimiter=',')

    # Save nodes list
    nodes_data = list(G.nodes.data())  # [(node_name, {attr1, attr2}),...]
    print(nodes_data)
    with open(os.path.join(dst_dir, 'task_graph_X.csv'), 'w') as f:
        print('node_name/poi_id,checkin_cnt,poi_catid,latitude,longitude', file=f)
        for each in nodes_data:
            node_name = each[0]
            checkin_cnt = each[1]['checkin_cnt']
            poi_catid = each[1]['poi_catid']
            latitude = each[1]['latitude']
            longitude = each[1]['longitude']
            print(f'{node_name},{checkin_cnt},'
                  f'{poi_catid},'
                  f'{latitude},{longitude}', file=f)


def build_global_worker_social_graph(train_df, social_df, worker_df):

    train_workers = set(train_df['user_id'].tolist())

    G = nx.DiGraph()
    # Add nodes
    for i, row in worker_df.iterrows():
        node = row['user_id']
        if int(node) not in train_workers:
            continue
        if node not in G.nodes():
            G.add_node(row['user_id'],
                        checkin_cnt=1,
                        latitude=row['latitude'],
                        longitude=row['longitude'])
        else:
            G.nodes[node]['checkin_cnt'] = 1
    print(len(G.nodes))

    # Add edges
    for i, row in social_df.iterrows():
        if int(row['user1']) not in train_workers:
            continue
        if int(row['user2']) not in train_workers:
            continue

        if G.has_edge(row['user1'], row['user2']):
            G.edges[row['user1'], row['user2']]['weight'] = 1
            # import pdb; pdb.set_trace()
        else:
            G.add_edge(row['user1'], row['user2'], weight=1)
            # import pdb; pdb.set_trace()
       
    return G



            
def save_worker_graph_to_csv(G, dst_dir):
    # Save graph to an adj matrix file and a nodes file
    # Adj matrix file: edge from row_idx to col_idx with weight; Rows and columns are ordered according to nodes file.
    # Nodes file: node_name/poi_id, node features (category, location); Same node order with adj matrix.

    # Save adj matrix
    nodelist = G.nodes()
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    print(np.array(A))
    # np.save(os.path.join(dst_dir, 'adj_mtx.npy'), A.todense())
    np.savetxt(os.path.join(dst_dir, 'worker_graph_A.csv'), A.todense(), delimiter=',')

    # Save nodes list
    nodes_data = list(G.nodes.data())  # [(node_name, {attr1, attr2}),...]

    with open(os.path.join(dst_dir, 'worker_graph_X.csv'), 'w') as f:
        print('node_name/worker_id,latitude,longitude', file=f)
        for each in nodes_data:
            try:
                node_name = each[0]
                latitude = each[1]['latitude']
                longitude = each[1]['longitude']
            except:
                import pdb; pdb.set_trace()
            print(f'{node_name},'
                f'{latitude},{longitude}', file=f)

def save_graph_to_pickle(G, dst_dir):
    pickle.dump(G, open(os.path.join(dst_dir, 'graph.pkl'), 'wb'))


def save_graph_edgelist(G, dst_dir):
    nodelist = G.nodes()
    node_id2idx = {k: v for v, k in enumerate(nodelist)}

    with open(os.path.join(dst_dir, 'graph_node_id2idx.txt'), 'w') as f:
        for i, node in enumerate(nodelist):
            print(f'{node}, {i}', file=f)

    with open(os.path.join(dst_dir, 'graph_edge.edgelist'), 'w') as f:
        for edge in nx.generate_edgelist(G, data=['weight']):
            src_node, dst_node, weight = edge.split(' ')
            print(f'{node_id2idx[src_node]} {node_id2idx[dst_node]} {weight}', file=f)


def load_graph_adj_mtx(path):
    """A.shape: (num_node, num_node), edge from row_index to col_index with weight"""
    A = np.loadtxt(path, delimiter=',')
    return A


def load_graph_node_features(path, feature1='checkin_cnt', feature2='poi_catid_code',
                             feature3='latitude', feature4='longitude'):
    """X.shape: (num_node, 4), four features: checkin cnt, poi cat, latitude, longitude"""
    df = pd.read_csv(path)
    rlt_df = df[[feature1, feature2, feature3, feature4]]
    X = rlt_df.to_numpy()

    return X


def print_graph_statisics(G):
    print(f"Num of nodes: {G.number_of_nodes()}")
    print(f"Num of edges: {G.number_of_edges()}")

    # Node degrees (mean and percentiles)
    node_degrees = [each[1] for each in G.degree]
    print(f"Node degree (mean): {np.mean(node_degrees):.2f}")
    for i in range(0, 101, 20):
        print(f"Node degree ({i} percentile): {np.percentile(node_degrees, i)}")

    # Edge weights (mean and percentiles)
    edge_weights = []
    for n, nbrs in G.adj.items():
        for nbr, attr in nbrs.items():
            weight = attr['weight']
            edge_weights.append(weight)
    print(f"Edge frequency (mean): {np.mean(edge_weights):.2f}")
    for i in range(0, 101, 20):
        print(f"Edge frequency ({i} percentile): {np.percentile(edge_weights, i)}")


if __name__ == '__main__':
    dst_dir = r'data-process/Foursquare'
    DATASET = 'Foursquare'

    # dst_dir = r'data-process/Yelp_FGRec'
    # DATASET = 'Yelp'

    # Build task checkin trajectory graph
    train_df = pd.read_csv(os.path.join(dst_dir, DATASET+'_train.txt'))
    print('Build global task graph -----------------------------------')
    task_traj_G = build_global_task_checkin_graph(train_df)
    save_task_graph_to_csv(task_traj_G, dst_dir=dst_dir)

    # Build golbal worker social graph
    social_df = pd.read_csv(os.path.join(dst_dir, DATASET+'_social_relations.txt'))
    worker_df = pd.read_csv(os.path.join(dst_dir, DATASET+'_user_home.txt'))
    print('Build golbal worker social graph ----------------------------------')
    worker_social_G = build_global_worker_social_graph(train_df, social_df, worker_df)
    save_worker_graph_to_csv(worker_social_G, dst_dir=dst_dir)

    # Save graph to disk
    # save_graph_to_pickle(G, dst_dir=dst_dir)
    # save_graph_edgelist(G, dst_dir=dst_dir)
    print_graph_statisics(worker_social_G)
