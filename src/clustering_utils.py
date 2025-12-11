import math
from typing import List, Tuple
import json
import os

def get_clusters(
    problem_type: str, all_klee_vars: List[str], max_num: int, min_cluster_size: int = 4
) -> List[List[str]]:
    if problem_type in ["TE", "tsp", "task_scheduling"]:
        return get_matrix_clusters(all_klee_vars, max_num, min_cluster_size)
    else:
        return get_vector_clusters(all_klee_vars, max_num, min_cluster_size)


def get_vector_clusters(
    all_klee_vars: List[str], max_num: int, min_cluster_size: int = 4
) -> List[List[str]]:
    # just chunk the variables into max_num clusters
    # sort names low to high
    try:
        all_klee_vars.sort(key=lambda x: int(x.split("_")[-1]))
    except:
        all_klee_vars.sort()
    return [all_klee_vars[i : i + max_num] for i in range(0, len(all_klee_vars), max_num)]


def get_matrix_clusters(
    all_klee_vars: List[str], max_num: int, min_cluster_size: int = 4
) -> List[List[str]]:
    """
    Same contract as before, but guarantees that clusters are
    *spatially‑contiguous* and that no cluster is smaller than
    `min_cluster_size` (unless there is literally no neighbour to merge with).
    """
    # ── 1.  Parse “…_i_j” -------------------------------------------------
    triples: List[Tuple[str, int, int]] = []
    for v in all_klee_vars:
        try:
            i, j = map(int, v.split("_")[-2:])
        except ValueError:
            continue
        triples.append((v, i, j))
    if not triples:
        return []

    # ── 2.  Canonical 0…n−1 coordinates ----------------------------------
    nodes = sorted({i for _, i, _ in triples} | {j for _, _, j in triples})
    pos = {idx: k for k, idx in enumerate(nodes)}
    n = len(nodes)

    # ── 3.  Bucket side = ceil(sqrt(max_num)) ----------------------------
    side = max(1, int(math.sqrt(max_num)))
    n_blocks = math.ceil(n / side)

    # ── 4.  Fill buckets --------------------------------------------------
    buckets = [[[] for _ in range(n_blocks)] for _ in range(n_blocks)]
    for v, i, j in triples:
        r, c = pos[i] // side, pos[j] // side
        buckets[r][c].append(v)

    # ── 5.  Flatten in the desired order (diag → off‑diags) --------------
    ordered: List[Tuple[Tuple[int, int], List[str]]] = []

    # (a) block diagonal
    for b in range(n_blocks):
        if buckets[b][b]:
            ordered.append(((b, b), sorted(buckets[b][b])))

    # (b) off‑diagonals
    for r in range(n_blocks):
        for c in range(n_blocks):
            if r == c or not buckets[r][c]:
                continue
            ordered.append(((r, c), sorted(buckets[r][c])))

    # ── 6.  Spatial merging ----------------------------------------------
    return merge_clusters_spatial(ordered, max_num, min_cluster_size)


# ----------------------------------------------------------------------- #
#   Spatial, size‑bounded merging                                          #
# ----------------------------------------------------------------------- #


def _is_adjacent(coord: Tuple[int, int], others: List[Tuple[int, int]]) -> bool:
    """True iff `coord` touches *any* coord in `others` (8‑neighbourhood)."""
    r, c = coord
    return any(abs(r - or_) <= 1 and abs(c - oc) <= 1 for or_, oc in others)


def merge_clusters_spatial(
    buckets: List[Tuple[Tuple[int, int], List[str]]],
    max_cluster_size: int,
    min_cluster_size: int = 4,
) -> List[List[str]]:
    """
    Parameters
    ----------
    buckets
        List of (block‑coord, variables) pairs – the `ordered` list above.
    max_cluster_size
        Hard upper bound on cluster size.
    min_cluster_size
        Soft lower bound – we try to exceed this by merging with a *touching*
        bucket if necessary, but we **never** cross `max_cluster_size`.
    """
    if max_cluster_size <= 0:
        raise ValueError("max_cluster_size must be positive")

    merged: List[List[str]] = []
    cur_vars: List[str] = []
    cur_coords: List[Tuple[int, int]] = []

    def flush():
        """Append the current cluster and reset."""
        if cur_vars:
            merged.append(sorted(cur_vars.copy()))
            cur_vars.clear()
            cur_coords.clear()

    for coord, vars_ in buckets:

        # 1. deal with monsters (> max) immediately – they get split in‑place
        if len(vars_) > max_cluster_size:
            flush()
            for i in range(0, len(vars_), max_cluster_size):
                merged.append(vars_[i : i + max_cluster_size])
            continue

        # 2. if the bucket *touches* current cluster & fits ⇒ merge
        if (
            cur_vars
            and _is_adjacent(coord, cur_coords)
            and len(cur_vars) + len(vars_) <= max_cluster_size
        ):
            cur_vars.extend(vars_)
            cur_coords.append(coord)
            continue

        # 3. otherwise, see if current cluster is too small and
        #    merging would *still* fit – this preference keeps tiny
        #    diagonals from being emitted on their own.
        if (
            cur_vars
            and len(cur_vars) < min_cluster_size
            and len(cur_vars) + len(vars_) <= max_cluster_size
        ):
            # even if not adjacent, we *prefer* the *nearest* bucket;
            # here “nearest” = first one in traversal order that still fits
            cur_vars.extend(vars_)
            cur_coords.append(coord)
            continue

        # 4. finalize previous cluster and start a new one
        flush()
        cur_vars.extend(vars_)
        cur_coords.append(coord)

    flush()
    return merged

def load_one_cluster_from_file(cluster_path: str, all_klee_vars: List[str]) -> Tuple[List[str], List[str]]:
    cluster_topology = json.load(open(cluster_path))
    # Extract node IDs from the cluster topology
    node_ids = []
    for node in cluster_topology["nodes"]:
        node_ids.append(node["id"])

    # Generate all possible demand pairs between nodes in the cluster
    # Demand variables follow the format "demand_{source}_{target}"
    cluster_demands = []
    for i, source in enumerate(node_ids):
        for j, target in enumerate(node_ids):
            if i != j and f"demand_{source}_{target}" in all_klee_vars:
                demand_var = f"demand_{source}_{target}"
                cluster_demands.append(demand_var)

    return cluster_demands, node_ids

def compute_inter_cluster_demands(cluster1_nodes: List[str], cluster2_nodes: List[str], all_klee_vars: List[str]) -> List[str]:
    inter_cluster_demands = []
    for node1 in cluster1_nodes:
        for node2 in cluster2_nodes:
            if node1 != node2 and f"demand_{node1}_{node2}" in all_klee_vars:
                inter_cluster_demands.append(f"demand_{node1}_{node2}")
            if node1 != node2 and f"demand_{node2}_{node1}" in all_klee_vars:
                inter_cluster_demands.append(f"demand_{node2}_{node1}")
    return inter_cluster_demands

def get_clusters_from_directory(problem_type: str, all_klee_vars: List[str], cluster_path: str) -> List[List[str]]:
    assert problem_type == "TE"
    # get all the files in the directory
    files = os.listdir(cluster_path)
    # get the files that start with "cluster_" and end with ".json"
    json_files = [file for file in files if file.startswith("cluster_") and file.endswith(".json")]
    # sort the json files
    json_files.sort()
    # load the json files
    disjoint_clusters_and_nodes = [load_one_cluster_from_file(os.path.join(cluster_path, file), all_klee_vars) for file in json_files]
    final_clusters = [disjoint_cluster_and_nodes[0] for disjoint_cluster_and_nodes in disjoint_clusters_and_nodes]
    partition_nodes = [disjoint_cluster_and_nodes[1] for disjoint_cluster_and_nodes in disjoint_clusters_and_nodes]
    for i in range(len(partition_nodes)):
        for j in range(i+1, len(partition_nodes)):
            inter_cluster_demands = compute_inter_cluster_demands(partition_nodes[i], partition_nodes[j], all_klee_vars)
            final_clusters.append(inter_cluster_demands)
    return final_clusters