
import argparse
import networkx as nx
import polars as pl
import time
import sqlite3
import tqdm
import os

from networkx.algorithms.approximation import (min_weighted_vertex_cover, all_pairs_node_connectivity, node_connectivity, 
    maximum_independent_set, max_clique, clique_removal, large_clique_size, treewidth_min_degree, treewidth_min_fill_in, 
    min_weighted_vertex_cover, randomized_partitioning, one_exchange)
from networkx.algorithms.tree import is_tree as nx_is_tree
from networkx.algorithms.assortativity import average_neighbor_degree
from networkx.algorithms.asteroidal import is_at_free
from networkx.algorithms.bipartite import spectral_bipartivity
from networkx.algorithms.bridges import has_bridges
from networkx.algorithms.connectivity import all_node_cuts
from networkx.algorithms.covering import min_edge_cover
from networkx.algorithms.chordal import (is_chordal, chordal_graph_cliques, chordal_graph_treewidth)
from networkx.algorithms import (clustering, average_clustering,diameter, eccentricity, radius, dominating_set, 
                                 average_shortest_path_length, wiener_index, is_planar)
from networkx.algorithms.community import k_clique_communities

from car import (
    read_graphs, find_claws, find_C4_cycles, is_complete_bipartite,
    is_cycle_graph, is_complete_graph, is_path_graph, serialize_graph, 
    flatten_dict, stringify_dict, insert_data, summarize, 
    check_column_exists, find_missing_cells
)

# main is for creating a new local (tmp) db (and checking missing cols?), if we want to check for differences in current (old) 
# then run updatedbd.py
def main():

    # Read graphs
    graphs = read_graphs(graph_file_paths)
    graph_properties_list = []
    start_processing = time.time()
    print("Processing graphs...")
    for i, G in enumerate(tqdm.tqdm(graphs)):
        if nx_is_tree(G) or is_cycle_graph(G) or is_complete_graph(G) or is_path_graph(G) or is_complete_bipartite(G) or not find_claws(G) or not find_C4_cycles(G):
            continue

        try:
            all_pairs_conn = all_pairs_node_connectivity(G)
            min_conn = min(min(d.values()) for d in all_pairs_conn.values())
            max_conn = max(max(d.values()) for d in all_pairs_conn.values())

            ecc = eccentricity(G)
            min_ecc = min(ecc.values())
            max_ecc = max(ecc.values())

            cluster_coeffs = clustering(G)
            min_cluster_coeff = min(cluster_coeffs.values())
            max_cluster_coeff = max(cluster_coeffs.values())

            graph_properties = {
                "Graph number": i,
                "Graph": serialize_graph(G),
                "Cop Number": None,
                "Meyniel Satisfiable": None,
                "Num of nodes": G.number_of_nodes(),
                "Num of edges": G.number_of_edges(),
                "Approximation for node connectivity": node_connectivity(G),
                "Minimum all pairs node connectivity": min_conn,
                "Maximum all pairs node connectivity": max_conn,
                "Maximum independent set size": len(maximum_independent_set(G)),
                "Maximum clique size": len(max_clique(G)),
                "Largest clique size": large_clique_size(G),
                "Treewidth using minimum degree heuristic": treewidth_min_degree(G)[0],
                "Treewidth using minimum fill-in heuristic": treewidth_min_fill_in(G)[0],
                "Min weighted vertex cover size": len(min_weighted_vertex_cover(G)),
                "Average neighbor degree": average_neighbor_degree(G),
                "AT free": is_at_free(G),
                "Spectral bipartivity": spectral_bipartivity(G),
                "Does graph have bridges?": has_bridges(G),
                "Minimum clustering coefficient": min_cluster_coeff,
                "Maximum clustering coefficient": max_cluster_coeff,
                "Average clustering coefficient": average_clustering(G),
                "Diameter of the graph": diameter(G),
                "Minimum eccentricity": min_ecc,
                "Maximum eccentricity": max_ecc,
                "Radius of the graph": radius(G),
                "Domination number": len(dominating_set(G)),
                "Planar graph": is_planar(G),
                "Average shortest path length": average_shortest_path_length(G),
                "Wiener index": wiener_index(G),
                "Density": nx.density(G),
                "Chordal graph": is_chordal(G),
            }

            if is_chordal(G):
                graph_properties.update({
                    "Chordal graph treewidth": chordal_graph_treewidth(G),
                    "Chordal graph cliques": [list(clique) for clique in chordal_graph_cliques(G)],
                })
            else:
                graph_properties.update({
                    "Chordal graph treewidth": None,
                    "Chordal graph cliques": None,
                })

            graph_properties["K-clique communities"] = [list(community) for community in k_clique_communities(G, k=3)]

            node_cuts = list(all_node_cuts(G))
            for idx, cut in enumerate(node_cuts):
                graph_properties[f"Node cut {idx + 1}"] = list(cut)
            
            graph_properties.update({
                "Clique removal": clique_removal(G),
                "Randomized partitioning": randomized_partitioning(G),
                "One exchange": one_exchange(G),
                "Minimum edge cover": min_edge_cover(G),
            })

        except Exception as e:
            print(f"Error processing graph {i}: {e}")
            continue

        graph_properties_list.append(graph_properties)

    end_processing = time.time()
    processing_time = start_processing - end_processing
    print(f"\n\tProcessing of graphs took {processing_time:.2f} seconds.\n")
    
    db_conn_start_time = time.time()
        
    # Create a Polars DataFrame
    df = pl.DataFrame([stringify_dict(flatten_dict(d)) for d in graph_properties_list])

    # SQLite database filename
    db_filename = 'graph_properties.db'

    with sqlite3.connect(db_filename, timeout=30) as conn:
        cursor = conn.cursor()

        # Check if the table exists locally
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='graph_properties';")
        table_exists = cursor.fetchone() is not None

        if table_exists:
            # Drop the existing table if it exists
            cursor.execute("DROP TABLE graph_properties")

        # Create table
        columns = df.columns
        column_definitions = ", ".join([f'"{col}" TEXT' for col in columns])
        cursor.execute(f"CREATE TABLE IF NOT EXISTS graph_properties ({column_definitions})")
            
        # Insert data
        insert_data(cursor, df.iter_rows())
            
        conn.commit()  # Ensure changes are committed
        print("Data has been written to 'graph_properties.db'")

    # Measure end time for database connection
    db_conn_end_time = time.time()
    db_conn_time = db_conn_end_time - db_conn_start_time
    print(f"\n\tProcessing of graphs took {db_conn_time:.2f} seconds.\n")

    os.startfile(db_filename)
    
    column_name = input("Enter the column name to check for missing cells: ")

    # Check if the column exists in the table
    if check_column_exists(db_filename, column_name):
        find_missing_cells(db_filename, column_name)
    else:
        print(f"The column '{column_name}' does not exist in the table.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some graphs.')
    parser.add_argument('--graph_file_paths', nargs='+', required=True, help='Paths to the graph files')
    args = parser.parse_args()
    graph_file_paths = args.graph_file_paths
    main()
