import networkx as nx
import numpy as np
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sn

from parse_and_convert_graphml import *

def read_graph_xml(fname):
    assert fname.endswith('.xml')
    with open(fname, 'r') as f:
        data = f.read()
    xml_data = BeautifulSoup(data, "xml")
    devices = xml_data.find_all('Device')
    devices = [x.get('Name') for x in devices]
    demand_groups = [set(), set()]
    others = []
    for device in devices:
        if device.startswith("ibr"):
            demand_groups[0].add(device)
        elif device.endswith("SW"):
            demand_groups[1].add(device)
        else:
            others.append(device)

    G = nx.DiGraph()
    [G.add_node(devices[i]) for i in range(len(devices))]
    edges = xml_data.find_all('Link')
    min_link_cap = np.inf
    max_link_cap = 0
    avg_link_cap = 0
    list_link_cap = []
    for x in edges:
        src = x.get('StartDevice')
        dst = x.get('EndDevice')
        cap = float(x.get('Bandwidth'))
        if cap <= 0:
            continue
        G.add_edge(src, dst, capacity=cap)
        G.add_edge(dst, src, capacity=cap)
        min_link_cap = np.minimum(min_link_cap, cap)
        max_link_cap = np.maximum(max_link_cap, cap)
        avg_link_cap += cap
        list_link_cap.append(cap)
    avg_link_cap /= len(edges)
    sn.ecdfplot(list_link_cap)
    plt.xlabel("Capacity (Mbps)")
    plt.ylabel("Fraction of links")
    plt.show()
    h = []

    for scc_ids in nx.strongly_connected_components(G):
        scc = G.subgraph(scc_ids)
        if len(scc) > len(h):
            h = scc

    device_to_nid_mapping = dict()
    renamed_graph = nx.convert_node_labels_to_integers(h, label_attribute="device_name")
    nid_to_device_mapping = nx.get_node_attributes(renamed_graph, "device_name")
    for nid, device in nid_to_device_mapping.items():
        device_to_nid_mapping[device] = nid
    # for nid, node in enumerate(h.nodes()):
    #     new_graph.add_node(nid)
    #     revised_device_id_to_name_mapping[node] = nid
    #
    # for edge in h.edges():
    #     u = revised_device_id_to_name_mapping[edge[0]]
    #     v = revised_device_id_to_name_mapping[edge[1]]
    #     cap = edge['capacity']
    #     new_graph.add_edge(u, v, capacity=cap)
    #     new_graph.add_edge(v, u, capacity=cap)
    print("==== topology stat ====")
    print(f"min link capacity: {min_link_cap}")
    print(f"max link capacity: {max_link_cap}")
    print(f"avg link capacity: {avg_link_cap}")
    print(f"sum all link capacities: {np.sum(list_link_cap)}")
    return renamed_graph, device_to_nid_mapping, min_link_cap, demand_groups


# topo_fpath_list = [
#     # "GtsCe", 
#     # "Cogentco",
#     # "Kdl",
#     ("../../OneWanMaxMinFairness/onewan/SWANTETopology.xml", "SWANTETopology"),
# ]
# for topo_fpath, topo_name in topo_fpath_list:
#     G = read_graph_xml(topo_fpath)[0]
#     fname = f'./{topo_name}.json'
#     write_graph_json(G, fname)