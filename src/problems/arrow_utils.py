from __future__ import annotations
import sys
import os
# Add parent directory to path for utils and common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
import time
from utils import *
from ortools.linear_solver import pywraplp
from .problem import Problem
from common import LAMBDA_MAX_VALUE
from ortools.linear_solver import pywraplp
from typing import List, Dict, Set, Tuple, Optional, Any
import heapq
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Dict, List, Tuple, DefaultDict, Optional, Any
from collections import defaultdict, Counter

ENABLE_PRINT = False
WAVELENGTH_CAPACITY = 100


class Edge:
    """A directed edge in the traffic engineering network with capacity constraints."""

    def __init__(self, source: str, target: str, capacity: float):
        self.source = source
        self.target = target
        self.capacity = float(capacity)
        self._flow = float(0)

    @property
    def flow(self) -> float:
        return self._flow

    @flow.setter
    def flow(self, value: float) -> None:
        self._flow = float(value)

    @property
    def remaining_capacity(self) -> float:
        """Calculate remaining capacity on this edge."""
        return self.capacity - self.flow

    def can_add_flow(self, additional_flow: float) -> bool:
        """Check if additional flow can be added to this edge."""
        return self.remaining_capacity >= additional_flow

    def add_flow(self, additional_flow: float) -> bool:
        """Add flow to this edge if possible."""
        if self.can_add_flow(additional_flow):
            self.flow += additional_flow
            return True
        return False

    def __repr__(self) -> str:
        return (
            f"Edge({self.source}->{self.target}, cap={self.capacity}, flow={self.flow})"
        )


class Fiber:
    def __init__(self, fiber_id, fiber_ip_path, num_wave):
        self.fiber_id = int(fiber_id)
        self.fiber_ip_path: list[str] = fiber_ip_path.split("-")
        self.num_wave: int = int(num_wave)

    def __repr__(self) -> str:
        return f"Fiber({self.fiber_id}, {self.fiber_ip_path}, {self.num_wave})"


class OpticalTopology:
    def __init__(self, fiber_dict: Dict[int, Fiber]):
        self.fiber_dict = fiber_dict

    def remove_fiber(self, fiber_id: str):
        if fiber_id in self.fiber_dict:
            del self.fiber_dict[fiber_id]

    def get_nodes(self, fiber_dict: Dict[int, Fiber]) -> List[str]:
        nodes = set()
        for fiber in fiber_dict.values():
            nodes.update(fiber.get_nodes())
        return list(nodes)


@dataclass
class LotteryTicket:
    """
    A ticket maps each *IP-layer Edge* to one or more optical Fibers and
    says how many spare wavelengths each Fiber contributes.

        allocations : List[(edge, [(fiber, waves), ... ])]

    Example (chain A-B, B-C):

        allocations = [
            (Edge(A,B), [(fiber_AB, 3)]),             # 3 λ on span A-B
            (Edge(B,C), [(fiber_BC, 2)])              # 2 λ on span B-C
        ]
    """

    def __init__(
        self, ticket_id: str, allocations: List[Tuple[Edge, List[Tuple[Fiber, int]]]]
    ):
        self.ticket_id: str = ticket_id
        self.allocations: List[Tuple[Edge, List[Tuple[Fiber, int]]]] = allocations

    def remove_zero_edges(self) -> None:
        """Remove edges with zero capacity."""
        self.allocations = [
            (edge, fiber_allocations) for edge, fiber_allocations in self.allocations if edge.capacity > 0
        ]

    def fix_edge_capacities(self) -> None:
        """Fix edge capacities based on fiber allocations."""
        for edge, fiber_allocations in self.allocations:
            # Calculate capacity using proper multi-hop path analysis
            capacity = get_capacity_from_fiber_allocations(
                fiber_allocations, edge.source, edge.target
            )
            edge.capacity = capacity

    def copy(self) -> 'LotteryTicket':
        """Create a deep copy of this lottery ticket."""
        from copy import deepcopy
        new_allocations = []
        for edge, fiber_allocations in self.allocations:
            # Create a new edge with the same properties
            new_edge = Edge(edge.source, edge.target, edge.capacity)
            # Create new fiber allocations (fibers are immutable, so we can reuse them)
            new_fiber_allocations = [(fiber, wave) for fiber, wave in fiber_allocations]
            new_allocations.append((new_edge, new_fiber_allocations))
        return LotteryTicket(self.ticket_id, new_allocations)

    def __repr__(self) -> str:
        return f"LotteryTicket({self.ticket_id}, {self.allocations})"


def get_capacity_from_fiber_allocations(
    fiber_allocations: List[Tuple[Fiber, int]], source: str, target: str
) -> int:
    """
    Calculate the capacity for a multi-hop edge based on fiber allocations.

    The capacity is the sum of all possible paths from source to target,
    where each path's capacity is limited by its bottleneck hop.

    Examples:
    - EdgeA-D:: A-D:1 , A-D:1 -> capacity = 200 (direct path: 1+1=2 wavelengths)
    - EdgeA-D:: A-D:1 , B-D:1 -> capacity = 100 (only A-D fiber helps A->D)
    - EdgeA-D:: A-B:2 , B-D:1, B-C:1, C-D:1 -> capacity = 200 (A->B->D: 1, A->B->C->D: 1)
    """
    from collections import defaultdict

    # Group fiber allocations by the fiber's path and sum wavelengths for same paths
    hop_wavelengths = defaultdict(int)
    for fiber, wavelengths in fiber_allocations:
        path = tuple(fiber.fiber_ip_path)  # e.g., ['A', 'B'] -> ('A', 'B')
        hop_wavelengths[path] += wavelengths

    # Find all possible paths from source to target using the available fibers
    def find_paths_recursive(
        current_node, target_node, visited, current_path, all_paths
    ):
        if current_node == target_node:
            all_paths.append(current_path[:])
            return

        visited.add(current_node)
        # Look for hops that start from current_node
        for hop in hop_wavelengths.keys():
            if len(hop) == 2 and hop[0] == current_node and hop[1] not in visited:
                current_path.append(hop)
                find_paths_recursive(
                    hop[1], target_node, visited, current_path, all_paths
                )
                current_path.pop()
        visited.remove(current_node)

    # Find all possible paths
    all_paths = []
    find_paths_recursive(source, target, set(), [], all_paths)

    # Calculate capacity for each path (bottleneck) and sum them
    total_capacity = 0

    for path in all_paths:
        # Calculate bottleneck capacity for this path
        path_capacity = float("inf")
        for hop in path:
            hop_capacity = hop_wavelengths.get(hop, 0)
            path_capacity = min(path_capacity, hop_capacity)

        if path_capacity != float("inf") and path_capacity > 0:
            total_capacity += path_capacity

    return total_capacity * WAVELENGTH_CAPACITY


def ticket_to_demand_pinning_format(ticket: LotteryTicket):
    """
    Convert a lottery ticket to the format required by demand_pinning_TE.

    Args:
        ticket: LotteryTicket object containing edge allocations

    Returns:
        dict: Dictionary with 'num_nodes' and 'edges' keys formatted for demand_pinning_TE
    """
    # Extract all unique nodes from the ticket edges
    nodes = set()
    edge_capacities = {}

    for edge, fiber_allocations in ticket.allocations:
        nodes.add(edge.source)
        nodes.add(edge.target)
        # Calculate capacity using proper multi-hop path analysis
        capacity = get_capacity_from_fiber_allocations(
            fiber_allocations, edge.source, edge.target
        )
        edge_capacities[(edge.source, edge.target)] = capacity

    # Convert nodes to sorted list for consistent indexing
    node_list = sorted(list(nodes))
    node_to_index = {node: idx for idx, node in enumerate(node_list)}

    # Create edges dictionary with integer indices as expected by demand_pinning_TE
    edges = {}
    for (source_str, target_str), capacity in edge_capacities.items():
        source_idx = node_to_index[source_str]
        target_idx = node_to_index[target_str]
        edge_key = f"edge_{source_idx}_{target_idx}"
        edges[edge_key] = int(capacity)

    return {
        "num_nodes": len(node_list),
        "edges": edges,
        "node_mapping": node_to_index,  # Helpful for debugging/reference
        "node_list": node_list,  # Helpful for debugging/reference
    }
