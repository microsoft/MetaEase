import networkx as nx
import numpy as np
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sn
import parse_and_convert_xml
import pandas as pd
from collections import defaultdict


bin_names = {
    1: "<=2000", 
    2: ">2000", 
    3: ">5000", 
    4: ">10000"
}

def get_demand_bin_name(demand):
    if demand > 10000:
        return 4
    elif demand > 5000:
        return 1
    elif demand > 2000:
        return 1
    else:
        return 1

def get_demand(fname, device):
    with open(fname, 'r') as f:
        data = f.read()
        traffic_data = BeautifulSoup(data, "xml")
        tm_dict = {}
        demands = traffic_data.find_all('SwanDemand')

        for dem in demands:
            source = dem.get('SourceRouter')
            dest = dem.get('DestinationRouter')
            if source not in device or dest not in device:
                continue

            if source not in tm_dict:
                tm_dict[source] = {}
            if dest not in tm_dict[source]:
                tm_dict[source][dest] = 0
            tm_dict[source][dest] += float(dem.get('Mbps'))
        
        num_devices = len(device)
        tm = np.zeros(shape=(num_devices, num_devices))
        for source in tm_dict:
            for dest in tm_dict[source]:
                tm[device[source]][device[dest]] = tm_dict[source][dest]
        
        print("==== demand stat ====")
        print(f"shape: {tm.shape}")
        print(f"max demand: {np.max(tm)}")
        print(f"min demand: {np.min(tm)}")
        print(f"avg demand: {np.average(tm)}")
        print(f"sum demand: {np.sum(tm)}")
        print(f"min non-zero demand: {np.min(tm[tm > 0])}")
        print(f"avg non-zero demand: {np.average(tm[tm > 0])}")
        print(f"fraction of non-zeros {np.count_nonzero(tm) / (tm.shape[0] * tm.shape[1])}")
        print(f"fraction of 0 < demand < 10000", {np.count_nonzero(tm[(tm > 0) * (tm < 10000)]) / (tm.shape[0] * tm.shape[1])})
        print(f"num non-zero demands < 10000 divided by non-zero demands {np.count_nonzero(tm[(tm > 0) * (tm < 10000)]) / np.count_nonzero(tm > 0)}")
        print(f"total sum all demands >= 10000 {np.sum(tm[tm >= 10000])}")

        pinned_shortest_path_len_list = []
        all_shortest_path_len_list = []
        nonpinned_shortest_path_len_list = []
        demand_charac = defaultdict(int)
        total_num_demands = 0
        for src in tm_dict:
            for dst in tm_dict[src]:
                if tm_dict[src][dst] > 0:                
                    spl = nx.shortest_path_length(topo, device[src], device[dst])
                    demand_charac[(spl, get_demand_bin_name(tm_dict[src][dst]))] += 1
                    all_shortest_path_len_list.append(spl)
                    if tm_dict[src][dst] <= threshold:
                        pinned_shortest_path_len_list.append(spl)
                    else:
                        nonpinned_shortest_path_len_list.append(spl)
                    total_num_demands += 1
        print("==== path len stat ====")
        pinned_shortest_path_len_list = np.array(pinned_shortest_path_len_list)
        all_shortest_path_len_list = np.array(all_shortest_path_len_list)
        nonpinned_shortest_path_len_list = np.array(nonpinned_shortest_path_len_list)
        print("min pinned path len: ", np.min(pinned_shortest_path_len_list))
        print("max pinned path len: ", np.max(pinned_shortest_path_len_list))
        print("avg pinned path len: ", np.average(pinned_shortest_path_len_list))
        print("min non-pinned path len: ", np.min(nonpinned_shortest_path_len_list))
        print("max non-pinned path len: ", np.max(nonpinned_shortest_path_len_list))
        print("avg non-pinned path len: ", np.average(nonpinned_shortest_path_len_list))
        print("min all path len: ", np.min(all_shortest_path_len_list))
        print("max all path len: ", np.max(all_shortest_path_len_list))
        print("avg all path len: ", np.average(all_shortest_path_len_list))
        # sn.ecdfplot(pinned_shortest_path_len_list, label="pinned")
        # sn.ecdfplot(nonpinned_shortest_path_len_list, label="not pinned")
        # sn.ecdfplot(all_shortest_path_len_list, label="all")
        # plt.legend()
        # plt.show()

        dict_df = {"spl": [], "demand bin": [], "count": []}
        max_spl = np.max(all_shortest_path_len_list)
        for spl in range(1, max_spl + 1):
            for db in bin_names:
                dict_df["spl"].append(spl)
                dict_df["demand bin"].append(db)
                if (spl, db) in demand_charac:
                    dict_df["count"].append(demand_charac[spl, db] / total_num_demands)
                else:
                    dict_df["count"].append(0)
        df = pd.DataFrame(dict_df).pivot("spl", "demand bin", "count").fillna(0)
        print(df)
        print(df.sum(axis=0))
        print(df.sum(axis=1))
        sn.heatmap(df, xticklabels=[bin_names[i] for i in sorted(bin_names.keys())], cmap="Spectral")
        plt.ylabel("Shortest Path Len", fontsize=18)
        plt.xlabel("Flow Size", fontsize=18)
        plt.show()


topo_fname = "../../OneWanMaxMinFairness/onewan/SWANTETopology.xml"
traffic_fname = "../../OneWanMaxMinFairness/onewan/BackboneDemands.xml"
topo, device_to_nid_mapping, _, _ = parse_and_convert_xml.read_graph_xml(topo_fname)
threshold = 10000
get_demand(traffic_fname, device_to_nid_mapping)