"""
Arrow Problem Implementation with Parallel Processing Support

This module implements the Arrow problem with support for parallel processing of scenarios.
The parallel processing can be enabled/disabled using the ENABLE_PARALLEL_PROCESSING flag.

Key features:
- Parallel processing of optimal_wrapper, relaxed_optimal_wrapper, and arrow_wrapper
- Automatic fallback to sequential processing if parallel processing fails
- Configurable number of cores (uses min(cpu_count(), num_scenarios))
- Error handling and logging for debugging
"""

from .arrow_utils import *
from .programs_TE import *
import sys
import os
# Add parent directory to path to import common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import time
import os
import json
import random
from itertools import product
from typing import Dict, List, Optional, Tuple, Set
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial

ENABLE_PRINT = False
LOAD_TICKETS_FROM_FILE = True
ENABLE_PARALLEL_PROCESSING = True  # Set to False to disable parallel processing

# TODO: this can use documentation since this is not a standard CS problem, you can describe what the 
# Optimal is doing differently from what arrow does and what it means to be optimal in this comment.
# TODO: this seems like a standard multi-commodity flow problem, is not solving Arrow optimally? am I right? I believe what your doing is
# first finding the set of lottery tickets for the optimal and then using those computing the total flow you can achieve optimally under that setting.
# All of that context needs to be explained in a class documentation/file documentation.
def optimal_TE_for_arrow(num_nodes, edges, demands, possible_demands=None):
    # Create the solver
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        raise Exception("Solver could not be created!")
    all_vars = {}
    flow_on_path_vars = {}
    all_paths = {}
    nodes = range(num_nodes)
    for from_ in nodes:
        for to_ in nodes:
            if from_ != to_:
                paths = find_all_paths(
                    {"num_nodes": num_nodes, **edges}, str(from_), str(to_)
                )
                all_paths[(from_, to_)] = paths
                for path in paths:
                    path_string = "_".join(path)
                    var_name = f"aux_flowpath_{path_string}"
                    min_capacity_on_path = get_min_capacity_on_path(path, edges)
                    flow_on_path_vars[var_name] = solver.NumVar(
                        0, min_capacity_on_path, var_name
                    )

    # Create flow variables for
    if possible_demands is None:
        possible_demands, _ = find_possible_demands({"num_nodes": num_nodes, **edges})
    for from_, to_ in demands:
        all_vars[f"demand_{from_}_{to_}"] = demands[(from_, to_)]
    flow_pair_vars = {}
    flowpath_constraints = {}
    flow_demand_constraints = {}

    for from_, to_ in possible_demands:
        if (from_, to_) in demands:
            flow_pair_vars[f"aux_flow_{from_}_{to_}"] = solver.NumVar(
                0, 2 * demands[(from_, to_)], f"aux_flow_{from_}_{to_}"
            )
            flow_demand_constraints[(from_, to_)] = solver.Add(
                flow_pair_vars[f"aux_flow_{from_}_{to_}"] <= demands[(from_, to_)]
            )
            # the aux_flow_{from_}_{to_} is equal to the sum of the flow on all paths from from_ to to_
            paths = all_paths[(from_, to_)]
            path_names = ["_".join(path) for path in paths]
            sum_flow_on_paths = solver.Sum(
                flow_on_path_vars[f"aux_flowpath_{path_name}"]
                for path_name in path_names
            )
            constraint = solver.Add(
                flow_pair_vars[f"aux_flow_{from_}_{to_}"] == sum_flow_on_paths
            )
            flowpath_constraints[(from_, to_)] = constraint

    capacity_constraints = {}
    # Add capacity constraints for each edge
    for from_ in nodes:
        for to_ in nodes:
            if from_ != to_:
                edge = f"edge_{from_}_{to_}"
                if edge in edges:
                    capacity = edges[edge]
                    # find all the paths that use this edge
                    every_path = []
                    for key, value in all_paths.items():
                        every_path.extend("_".join(path) for path in value)
                    paths_with_this_edge = [
                        path for path in every_path if f"{from_}_{to_}" in path
                    ]
                    constraint = solver.Add(
                        solver.Sum(
                            flow_on_path_vars[f"aux_flowpath_{path}"]
                            for path in paths_with_this_edge
                        )
                        <= capacity
                    )
                    capacity_constraints[f"{from_}_{to_}"] = constraint

    # Objective: Maximize the total flow
    total_flow = solver.Sum(flow_pair_vars.values())
    solver.Maximize(total_flow)

    # Solve the problem
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise Exception("Solver did not find an optimal solution!")

    # Extract results
    flow_values = {
        edge: flow_pair_vars[edge].solution_value() for edge in flow_pair_vars
    }
    flowpath_values = {
        path: flow_on_path_vars[path].solution_value() for path in flow_on_path_vars
    }
    for key in flow_pair_vars:
        all_vars[key] = flow_pair_vars[key].solution_value()

    for key in flow_on_path_vars:
        all_vars[key] = flow_on_path_vars[key].solution_value()

    for key in flow_demand_constraints:
        all_vars[f"lambda_flow_demand_{key[0]}_{key[1]}"] = flow_demand_constraints[
            key
        ].dual_value()

    for key in flowpath_constraints:
        all_vars[f"lambda_flowpath_{key[0]}_{key[1]}"] = flowpath_constraints[
            key
        ].dual_value()

    for key in capacity_constraints:
        all_vars[f"lambda_capacity_{key}"] = capacity_constraints[key].dual_value()

    total_flow_value = total_flow.solution_value()
    return {
        "flow_values": flow_values,
        "flowpath_values": flowpath_values,
        "optimal_total_flow": total_flow_value,
        "all_vars": all_vars,
    }


# ---------------------------------------------------------------------
# Generate tickets using feasibility optimization
# ---------------------------------------------------------------------
# TODO: add documentaition such as:
# Calls generate_general_feasibility_tickets under the hood (which produces all mathematically valid ways to assign fiber wavelengths to logical edges.)
# This function wraps around that call to filter any solutions that are technically valid but useless...
def generate_all_tickets(
    fiber_dict: Dict[int, Fiber],
    num_tickets: Optional[int] = None,
) -> Tuple[List[LotteryTicket], List[str], Dict[int, Dict], List[Tuple[str, str]]]:
    """
    Generate all lottery tickets by solving a feasibility optimization problem.

    This function formulates and solves an integer programming problem:
    - Variables: x_{from}_{to} = capacity of edge from->to for all possible node pairs
    - Constraints: For each fiber, sum of all x_{from}_{to} traversing that fiber = fiber capacity
    - Solution: All feasible integer solutions that represent logical trade-offs

    General structure:
    x_{from}_{to}: capacity of the edge from->to for every possible from->to
    Set of linear constraints for each optical fiber where sum(all x_{from}_{to} traversing that fiber) = num_wavelength of that fiber

    Example for A-B(4), B-C(4) topology:
    Variables: x_A_B, x_B_C, x_A_C
    Constraints: x_A_B + x_A_C = 4, x_B_C + x_A_C = 4
    Solutions: (x_A_B, x_B_C, x_A_C) ∈ {(0,0,4), (1,1,3), (2,2,2), (3,3,1), (4,4,0)}
    """

    # Extract all nodes from fiber topology
    all_nodes = set()
    for fiber in fiber_dict.values():
        all_nodes.update(fiber.fiber_ip_path)

    node_list = sorted(all_nodes)

    # General implementation: works for ANY topology
    tickets, node_list, fiber_edge_constraints, all_edges = generate_general_feasibility_tickets(fiber_dict, node_list)
    tickets = filter_wasteful_tickets(tickets, fiber_dict)
    return tickets, node_list, fiber_edge_constraints, all_edges

# TODO: the documentation for this function can be improved to tell the reader what is the difference between this and the previous function in this file.
def generate_general_feasibility_tickets(
    fiber_dict: Dict[int, Fiber], node_list: List[str]
) -> Tuple[List[LotteryTicket], List[str], Dict[int, Dict], List[Tuple[str, str]]]:
    """
    Generate tickets using general feasibility optimization for ANY topology.

    This implements the general integer programming approach:
    - Variables: x_{from}_{to} for all possible node pairs
    - Constraints: For each fiber, sum of x_{from}_{to} traversing that fiber = fiber capacity
    - Solution: All feasible integer solutions that represent logical trade-offs
    """

    # Build fiber graph for path analysis
    fiber_graph = build_fiber_graph(fiber_dict)

    # Generate only valid edges that can be routed through the fiber topology
    all_edges = []
    edge_count = 0
    for i, from_node in enumerate(node_list):
        for j, to_node in enumerate(node_list):
            if from_node != to_node:
                edge_count += 1
                # Check if this edge can be routed through any fiber
                can_route = False
                for fiber in fiber_dict.values():
                    if edge_uses_fiber(from_node, to_node, fiber, fiber_graph):
                        can_route = True
                        break
                if can_route:
                    all_edges.append((from_node, to_node))

    print(f"Checked {edge_count} potential edges, found {len(all_edges)} routable edges")
    print(f"All possible edges: {all_edges}")

    if len(all_edges) == 0:
        print(f"ERROR: No routable edges found! This topology cannot support any traffic.")
        return [], node_list, {}, []

    # For each fiber, determine which edges traverse it
    fiber_edge_constraints = {}
    for fiber_id, fiber in fiber_dict.items():
        fiber_capacity = fiber.num_wave

        # Find all edges that use this fiber
        traversing_edges = []
        for from_node, to_node in all_edges:
            uses = edge_uses_fiber(from_node, to_node, fiber, fiber_graph)
            if uses:
                traversing_edges.append((from_node, to_node))

        fiber_edge_constraints[fiber_id] = {
            "capacity": fiber_capacity,
            "traversing_edges": traversing_edges,
        }

        print(
            f"Fiber {fiber_id} (capacity: {fiber_capacity}) supports edges: {traversing_edges}"
        )

    # Check if any fibers have no supporting edges
    fibers_with_no_edges = [fid for fid, info in fiber_edge_constraints.items() if len(info["traversing_edges"]) == 0]
    if fibers_with_no_edges:
        print(f"WARNING: Fibers {fibers_with_no_edges} support no edges - this may cause constraint satisfaction issues")

    # Use general constraint satisfaction to find all feasible solutions
    feasible_solutions = enumerate_general_solutions(fiber_edge_constraints, all_edges)

    print(f"Found {len(feasible_solutions)} feasible solutions")

    # Convert solutions to LotteryTicket objects
    tickets = []
    for i, solution in enumerate(feasible_solutions):
        allocations = solution_to_allocations(solution, fiber_dict)
        if ENABLE_PRINT:
            print(f"Solution {i}: {solution}")
            print(f"Allocations: {allocations}")
        if allocations:  # Only create ticket if there are allocations
            ticket = LotteryTicket(f"T{i+1}", allocations)
            # Don't call fix_edge_capacities() as it overrides the correct capacity values
            ticket.fix_edge_capacities()
            ticket.remove_zero_edges()
            tickets.append(ticket)

            # Create a description
            edge_descriptions = [
                f"{edge[0]}->{edge[1]}={capacity}λ"
                for edge, capacity in solution.items()
                if capacity > 0
            ]
            description = ", ".join(edge_descriptions) if edge_descriptions else "empty"
            if ENABLE_PRINT:
                print(f"Added T{i+1}: {description}")
        else:
            print(f"Solution {i}: {solution} is empty")

    print(f"Successfully created {len(tickets)} tickets")
    return tickets, node_list, fiber_edge_constraints, all_edges

# TODO: again, this function's documentation needs to be updated to describe when and how this function should be used.
# TODO: one thing that may be confusing is that this file is very different than those for knapsack, mwm, or other hueristics we have.
# It has much more constructs that are different. If your guiding someone to add a new heuristic to evaluate, how would you factor this in?
def enumerate_general_solutions(
    fiber_edge_constraints: Dict[int, Dict], all_edges: List[Tuple[str, str]]
) -> List[Dict[Tuple[str, str], int]]:
    """
    Enumerate solutions for any topology using proper SAT/constraint satisfaction.

    This implements a correct constraint satisfaction solver that:
    1. Sets up variables for edge capacities
    2. Creates constraints based on fiber capacity limits (equality constraints)
    3. Uses OR-Tools to find all feasible solutions
    4. Respects directed fibers (A-B can only carry A->B traffic)
    5. Aggregates fibers that support the same physical path
    """

    print("Using proper SAT/constraint satisfaction solver...")
    from ortools.sat.python import cp_model

    feasible_solutions = []

    # Get the maximum capacity from any fiber
    max_capacity = max(info["capacity"] for info in fiber_edge_constraints.values())
    print(f"Max capacity: {max_capacity}")

    # Create a CP-SAT model
    model = cp_model.CpModel()

    # Create variables for each edge capacity
    edge_vars = {}
    for edge in all_edges:
        edge_vars[edge] = model.NewIntVar(0, max_capacity, f"edge_{edge[0]}_{edge[1]}")

    # Aggregate fibers by their physical path to avoid conflicts
    # Group fibers that support the same physical path
    fiber_groups = {}
    for fiber_id, constraint_info in fiber_edge_constraints.items():
        # Create a key based on the physical path (sorted nodes)
        # For fibers that support the same edges, group them together
        traversing_edges = constraint_info["traversing_edges"]
        if traversing_edges:
            # Create a key based on the edges this fiber supports
            path_key = tuple(sorted(traversing_edges))
        else:
            path_key = ()

        if path_key not in fiber_groups:
            fiber_groups[path_key] = []
        fiber_groups[path_key].append((fiber_id, constraint_info))

    # Add constraints for each fiber group
    for path_key, fibers in fiber_groups.items():
        if not path_key:  # Skip empty paths
            continue

        # Get all edges that can use this fiber group
        edges_for_group = set()
        total_capacity = 0
        for fiber_id, constraint_info in fibers:
            edges_for_group.update(constraint_info["traversing_edges"])
            total_capacity += constraint_info["capacity"]

        # Constraint: sum of edge capacities using this fiber group <= total capacity
        # Changed from equality to inequality to make the problem feasible
        edge_vars_for_group = [
            edge_vars[edge] for edge in edges_for_group if edge in edge_vars
        ]
        if edge_vars_for_group:
            model.Add(sum(edge_vars_for_group) <= total_capacity)
            # model.Add(sum(edge_vars_for_group) > total_capacity - 1)
            if ENABLE_PRINT:
                print(
                    f"Added constraint for fiber group {path_key}: sum of {list(edges_for_group)} <= {total_capacity}"
                )
        else:
            print(f"  WARNING: No edge variables found for fiber group {path_key}")


    # Create a solution collector to find all solutions
    class SolutionCollector(cp_model.CpSolverSolutionCallback):
        def __init__(self, edge_vars, all_edges):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self._edge_vars = edge_vars
            self._all_edges = all_edges
            self.solutions = []

        def on_solution_callback(self):
            solution = {}
            for edge in self._all_edges:
                solution[edge] = self.Value(self._edge_vars[edge])
            self.solutions.append(solution)

    # Solve and collect all solutions
    print(f"Setting up solver...")
    solver = cp_model.CpSolver()
    solution_collector = SolutionCollector(edge_vars, all_edges)

    # Set parameters for finding all solutions
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.max_time_in_seconds = 30.0  # Limit search time

    # Solve the model
    print("Solving with OR-Tools SAT solver...")
    status = solver.Solve(model, solution_collector)

    print(f"Solver status: {status}")
    if status == cp_model.OPTIMAL:
        print(f"Solver found optimal solution")
    elif status == cp_model.FEASIBLE:
        print(f"Solver found feasible solution")
    elif status == cp_model.INFEASIBLE:
        print(f"Solver found problem INFEASIBLE - no solutions exist!")
        return []
    elif status == cp_model.MODEL_INVALID:
        print(f"Solver found model INVALID - constraint setup error!")
        return []
    else:
        print(f"Solver failed with unknown status: {status}")
        return []

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        all_solutions = solution_collector.solutions
        print(f"Found {len(all_solutions)} total solutions using OR-Tools")

        # Remove duplicate solutions
        unique_solutions = []
        seen_solution_signatures = set()

        for solution in all_solutions:
            # Create a signature based on non-zero edge values
            signature = []
            for edge, capacity in solution.items():
                if capacity > 0:
                    signature.append((edge, capacity))

            # Sort for consistent comparison
            signature = tuple(sorted(signature))

            if signature not in seen_solution_signatures:
                seen_solution_signatures.add(signature)
                unique_solutions.append(solution)

        feasible_solutions = unique_solutions
        if ENABLE_PRINT:
            print(
                f"Found {len(feasible_solutions)} unique feasible solutions after deduplication"
            )
    else:
        print(f"OR-Tools solver failed with status: {status}")

    print(f"Returning {len(feasible_solutions)} feasible solutions")
    return feasible_solutions


def filter_wasteful_tickets(
    tickets: List[LotteryTicket], 
    fiber_dict: Dict[int, Fiber]
) -> List[LotteryTicket]:
    """
    Filter out solutions that don't use any capacity of the topology.
    
    Keep all solutions that use at least some capacity of the topology.
    The constraint solver already ensures that fiber capacity constraints are satisfied.
    
    Args:
        tickets: List of tickets
        fiber_dict: Dictionary mapping fiber_id to Fiber objects
    
    Returns:
        List of solutions that use some topology capacity
    """
    # Keep all tickets that have at least some allocations
    new_tickets = []
    for ticket in tickets:
        if ticket.allocations:  # Keep tickets that have any allocations
            new_tickets.append(ticket)
    return new_tickets


def solution_to_allocations(
    solution: Dict[Tuple[str, str], int], fiber_dict: Dict[int, Fiber]
) -> List[Tuple[Edge, List[Tuple[Fiber, int]]]]:
    """Convert a solution (edge capacities) to lottery ticket allocations."""
    allocations = []

    # Build fiber graph for path analysis
    fiber_graph = build_fiber_graph(fiber_dict)

    for (from_node, to_node), capacity in solution.items():
        if capacity > 0:
            # Create edge with the exact capacity from the solution
            edge = Edge(
                from_node, to_node, capacity * WAVELENGTH_CAPACITY
            )  # Convert to WAVELENGTH_CAPACITY units

            # Find the actual path this edge would take
            fiber_path = find_fiber_path(from_node, to_node, fiber_dict)
            
            if fiber_path:
                # This is a multi-hop path, allocate capacity to all fibers in the path
                fiber_allocations = []
                
                # For multi-hop paths, we need to ensure all fibers along the path have sufficient capacity
                # The capacity should be distributed based on the bottleneck
                # For simplicity, allocate the full capacity to each fiber in the path
                # This ensures that the path can carry the required capacity
                for fiber in fiber_path:
                    fiber_allocations.append((fiber, capacity))
                
                allocations.append((edge, fiber_allocations))
            else:
                # Direct connection or single-hop path
                # Find all fibers that can support this edge using the existing logic
                supporting_fibers = []
                for fiber_id, fiber in fiber_dict.items():
                    if edge_uses_fiber(from_node, to_node, fiber, fiber_graph):
                        supporting_fibers.append(fiber)

                if supporting_fibers:
                    # Use the first supporting fiber as determined by the constraint solver
                    fiber_allocations = [(supporting_fibers[0], capacity)]
                    allocations.append((edge, fiber_allocations))

    return allocations


def find_fiber_path(
    from_node: str, to_node: str, fiber_dict: Dict[int, Fiber]
) -> List[Fiber]:
    """Find a sequence of fibers that connect from_node to to_node."""
    # Simple BFS to find fiber path
    from collections import deque

    # Build adjacency graph of fibers
    fiber_graph = {}
    for fiber_id, fiber in fiber_dict.items():
        path = fiber.fiber_ip_path
        for i in range(len(path) - 1):
            node_a, node_b = path[i], path[i + 1]
            if node_a not in fiber_graph:
                fiber_graph[node_a] = []
            if node_b not in fiber_graph:
                fiber_graph[node_b] = []
            fiber_graph[node_a].append((node_b, fiber))
            fiber_graph[node_b].append((node_a, fiber))

    # BFS to find path
    queue = deque([(from_node, [])])
    visited = set()

    while queue:
        current_node, path_fibers = queue.popleft()

        if current_node == to_node:
            return path_fibers

        if current_node in visited:
            continue
        visited.add(current_node)

        for next_node, fiber in fiber_graph.get(current_node, []):
            if next_node not in visited:
                queue.append((next_node, path_fibers + [fiber]))

    return []  # No path found


def build_fiber_graph(fiber_dict: Dict[int, Fiber]) -> Dict[str, List[Tuple[str, int]]]:
    """Build a directed adjacency graph where each node points to its neighbors and which fiber connects them."""
    graph = {}

    for fiber_id, fiber in fiber_dict.items():
        path = fiber.fiber_ip_path
        # Add directed edges for each fiber segment (fiber direction: path[i] -> path[i+1])
        for i in range(len(path) - 1):
            node_a, node_b = path[i], path[i + 1]

            if node_a not in graph:
                graph[node_a] = []
            if node_b not in graph:
                graph[node_b] = []

            # Only add edge in the direction of the fiber: node_a -> node_b
            graph[node_a].append((node_b, fiber_id))
            # Do NOT add the reverse edge: node_b -> node_a

    return graph


def edge_uses_fiber(
    from_node: str,
    to_node: str,
    fiber: Fiber,
    fiber_graph: Dict[str, List[Tuple[str, int]]],
) -> bool:
    """
    Determine if an edge from_node -> to_node would use the given fiber.

    For an edge to use a fiber, it must be part of a path that traverses that fiber.
    This includes:
    1. Direct connections along the fiber (respecting direction)
    2. Multi-hop paths that include this fiber as part of the route

    Examples for A-B-C topology:
    - Fiber A-B: supports A->B (direct) and A->C (multi-hop via A->B->C)
    - Fiber B-C: supports B->C (direct) and A->C (multi-hop via A->B->C)
    - Fiber A-B: does NOT support B->A or B->C (wrong direction)
    """
    fiber_path = fiber.fiber_ip_path

    # Check for direct connection: both nodes are adjacent in this fiber
    if from_node in fiber_path and to_node in fiber_path:
        from_idx = fiber_path.index(from_node)
        to_idx = fiber_path.index(to_node)

        # Direct connection in the right direction (fiber is directed)
        if abs(from_idx - to_idx) == 1 and from_idx < to_idx:
            return True

    # Check for multi-hop path: does any path from from_node to to_node use this fiber?
    # Use BFS to find if there's a path that goes through this fiber
    from collections import deque

    # Build the complete graph from all fibers (respecting direction)
    all_nodes = set()
    all_fiber_graph = {}

    # Get all fibers from the fiber_graph (we need to reconstruct the full topology)
    # fiber_graph maps node -> [(next_node, fiber_id)] and respects fiber direction
    for node, neighbors in fiber_graph.items():
        all_nodes.add(node)
        if node not in all_fiber_graph:
            all_fiber_graph[node] = []
        for next_node, _ in neighbors:
            all_nodes.add(next_node)
            all_fiber_graph[node].append(next_node)

    # BFS to find all paths from from_node to to_node
    queue = deque([(from_node, [from_node])])
    visited = set()

    while queue:
        current_node, path = queue.popleft()

        if current_node == to_node and len(path) > 1:
            # Found a path, check if it uses this fiber
            if path_uses_fiber(path, fiber):
                return True
            continue

        if current_node in visited:
            continue
        visited.add(current_node)

        # Limit path length to avoid over-constraining the system
        # Only allow paths of length 2-3 hops maximum for realistic optical networks
        if len(path) > 3:  # Maximum 3 hops
            continue

        for next_node in all_fiber_graph.get(current_node, []):
            if next_node not in path:  # Avoid cycles
                queue.append((next_node, path + [next_node]))

    return False


def path_uses_fiber(path: List[str], fiber: Fiber) -> bool:
    """Check if a path uses the given fiber (respecting fiber direction)."""
    fiber_path = fiber.fiber_ip_path

    # Check each consecutive pair in the path
    for i in range(len(path) - 1):
        from_node = path[i]
        to_node = path[i + 1]

        # Check if this hop is directly supported by this fiber
        if from_node in fiber_path and to_node in fiber_path:
            from_idx = fiber_path.index(from_node)
            to_idx = fiber_path.index(to_node)
            # Fiber is directed: must go in the same direction as the fiber
            if abs(from_idx - to_idx) == 1 and from_idx < to_idx:
                return True

    return False


def _process_optimal_scenario(args):
    """Helper function for parallel processing of optimal scenarios."""
    try:
        demands, all_tickets, self_instance = args
        return self_instance.optimal(demands, all_tickets)
    except Exception as e:
        print(f"Error in _process_optimal_scenario: {e}")
        raise e

def _process_relaxed_optimal_scenario(args):
    """Helper function for parallel processing of relaxed optimal scenarios."""
    try:
        demands, all_tickets, scenario, self_instance = args
        return self_instance.relaxed_optimal(demands, all_tickets)
    except Exception as e:
        print(f"Error in _process_relaxed_optimal_scenario for scenario: {scenario}")
        print(f"Error details: {e}")
        raise e

def _process_arrow_scenario(args):
    """Helper function for parallel processing of arrow scenarios."""
    try:
        demands, selected_tickets_per_seed, self_instance = args
        solutions = []
        for _, selected_tickets in selected_tickets_per_seed.items():
            solution = self_instance.arrow(demands, selected_tickets=selected_tickets)
            solutions.append(solution)
        return solutions
    except Exception as e:
        print(f"Error in _process_arrow_scenario: {e}")
        raise e

def _process_ticket_evaluation(args):
    """Helper function for parallel processing of ticket evaluation in optimal method."""
    try:
        ticket, demands = args
        ticket_id = ticket.ticket_id
        conv = ticket_to_demand_pinning_format(ticket)
        sol = optimal_TE_for_arrow(conv["num_nodes"], conv["edges"], demands)
        return (ticket_id, conv, sol)
    except Exception as e:
        print(f"Error in _process_ticket_evaluation for ticket {ticket.ticket_id}: {e}")
        raise e

def _copy_vars(
    dst: dict,
    flow_vals: Dict[str, float],
    path_vals: Dict[str, float],
    weight_vals: Optional[Dict[str, float]] = None,
    duals: Optional[Dict[str, float]] = None,
):
    """
    Push solver values into `dst` using the canonical names
       flow_<u>_<v>           total demand flow
       path_flow_<u>_<…>      per-path flow
       weight_<k>             ticket weights
    """

    # total-flow variables
    for k, v in flow_vals.items():
        if k.startswith("aux_flow_"):
            k = "flow_" + k[len("aux_flow_") :]
        dst[k] = float(v)

    # per-path variables
    for k, v in path_vals.items():
        if k.startswith("aux_flowpath_"):
            k = "path_flow_" + k[len("aux_flowpath_") :]
        dst[k] = float(v)

    # ticket weights
    if weight_vals:
        for k, w in weight_vals.items():
            dst[k] = float(w)

    # duals / shadow prices
    if duals:
        dst.update(duals)


class ArrowProblem(Problem):
    # TODO: the init function is massive and hard to follow, break it down and use helper functions.
    def __init__(self, problem_config_path):
        start_time = time.time()
        super().__init__(problem_config_path)
        topology_path = self.problem_config["topology_path"]
        self.max_num_optimal_tickets = self.problem_config.get("max_num_optimal_tickets", 100)
        self.fail_probability = self.problem_config.get("fail_probability", 0.002)
        self.num_scenarios = self.problem_config.get("num_scenarios", 10)
        self.num_tickets = self.problem_config["num_tickets"]
        self.num_random = self.problem_config.get("num_random", 10)
        self.seeds = [i for i in range(self.num_random)]
        print(f"Number of random runs for Arrow heuristic: {self.num_random}")
        fiber_dict = {}
        with open(topology_path, "r") as f:
            lines = f.readlines()
            for line_num, line in enumerate(lines):
                if "fiber_id" in line:
                    continue
                try:
                    fiber_id, fiber_ip_path, num_wave = line.strip().split(",")
                    fiber_dict[int(fiber_id)] = Fiber(fiber_id, fiber_ip_path, num_wave)
                except Exception as e:
                    print(f"  ERROR parsing line {line_num}: {line.strip()} - {e}")

        self.optical_topology = OpticalTopology(fiber_dict)

        fiber_ids = list(self.optical_topology.fiber_dict.keys())
        all_scenarios = [
            [x]
            for x in fiber_ids
        ]
        # shuffle the scenarios
        random.shuffle(all_scenarios)
        print(f"\033[92m[ArrowProblem] All scenarios: {all_scenarios}\033[0m")

        self.all_tickets_and_nodes_for_scenarios = []
        # generate all tickets and nodes for the scenarios
        topology_path_without_extension = topology_path.split(".")[0].replace("/", "_")
        topology_path_with_extension = topology_path_without_extension.replace(".", "_") + f"_{self.num_scenarios}scenarios_{self.num_tickets}tickets_{self.num_random}random"
        all_tickets_path = f"all_tickets_{topology_path_without_extension}.json"
        all_tickets_and_nodes_for_scenarios_path = f"all_tickets_and_nodes_for_scenarios_{topology_path_with_extension}.pkl"
        if (
            os.path.exists(all_tickets_and_nodes_for_scenarios_path)
            and LOAD_TICKETS_FROM_FILE
        ):
            with open(all_tickets_and_nodes_for_scenarios_path, "rb") as f:
                self.all_tickets_and_nodes_for_scenarios = pickle.load(f)
        else:
            print(f"Computing all tickets and nodes for scenarios (this may take a while)")
            for scenario in all_scenarios:
                all_tickets, nodes = self.get_all_new_tickets_and_nodes_for_scenario(
                    fiber_dict, scenario
                )
                if len(all_tickets) == 0:
                    print(f"No tickets found for scenario {scenario}")
                    continue
                self.all_tickets_and_nodes_for_scenarios.append(
                    (all_tickets, nodes, scenario)
                )
                if len(self.all_tickets_and_nodes_for_scenarios) >= self.num_scenarios:
                    break

            with open(all_tickets_and_nodes_for_scenarios_path, "wb") as f:
                pickle.dump(self.all_tickets_and_nodes_for_scenarios, f)

        scenarios = [scenario for _, _, scenario in self.all_tickets_and_nodes_for_scenarios]
        print(f"\033[92m[ArrowProblem] Selected scenarios: {scenarios}\033[0m")
        loading_start = time.time()
        if os.path.exists(all_tickets_path) and LOAD_TICKETS_FROM_FILE:
            print(f"Loading all tickets from {all_tickets_path}")
            with open(all_tickets_path, "r") as f:
                all_tickets_data = json.load(f)
            # Convert back from JSON-serializable format
            self.all_tickets = []
            for ticket_id, allocations_data in all_tickets_data.items():
                allocations = []
                for edge_data, fiber_allocations_data in allocations_data:
                    # Reconstruct Edge object
                    edge = Edge(
                        edge_data["source"], edge_data["target"], edge_data["capacity"]
                    )
                    # Reconstruct fiber allocations
                    fiber_allocations = []
                    for fiber_data, wave in fiber_allocations_data:
                        fiber = Fiber(
                            fiber_data["fiber_id"],
                            fiber_data["fiber_ip_path"],
                            fiber_data["num_wave"],
                        )
                        fiber_allocations.append((fiber, wave))
                    allocations.append((edge, fiber_allocations))
                self.all_tickets.append(LotteryTicket(ticket_id, allocations))
            # Extract all nodes from the topology, not from fiber IDs
            all_nodes = set()
            for fiber in fiber_dict.values():
                all_nodes.update(fiber.fiber_ip_path)
            self.all_nodes = sorted(all_nodes)
            # read pickle files
            with open(all_tickets_and_nodes_for_scenarios_path.replace("all_tickets_and_nodes_for_scenarios", "fiber_edge_constraints"), "rb") as f:
                self.fiber_edge_constraints = pickle.load(f)
            with open(all_tickets_and_nodes_for_scenarios_path.replace("all_tickets_and_nodes_for_scenarios", "all_edges"), "rb") as f:
                self.all_edges = pickle.load(f)
        else:
            print(f"Computing all tickets (this may take a while)")
            self.all_tickets, self.all_nodes, self.fiber_edge_constraints, self.all_edges = generate_all_tickets(fiber_dict)
            with open(all_tickets_and_nodes_for_scenarios_path.replace("all_tickets_and_nodes_for_scenarios", "fiber_edge_constraints"), "wb") as f:
                pickle.dump(self.fiber_edge_constraints, f)
            with open(all_tickets_and_nodes_for_scenarios_path.replace("all_tickets_and_nodes_for_scenarios", "all_edges"), "wb") as f:
                pickle.dump(self.all_edges, f)
            # Convert to JSON-serializable format
            tickets_data = {}
            for ticket in self.all_tickets:
                allocations_data = []
                for edge, fiber_allocations in ticket.allocations:
                    # Convert Edge to dict
                    edge_data = {
                        "source": edge.source,
                        "target": edge.target,
                        "capacity": edge.capacity,
                    }
                    # Convert fiber allocations to list of tuples with fiber dicts
                    fiber_allocations_data = []
                    for fiber, wave in fiber_allocations:
                        fiber_data = {
                            "fiber_id": fiber.fiber_id,
                            "fiber_ip_path": "-".join(fiber.fiber_ip_path),
                            "num_wave": fiber.num_wave,
                        }
                        fiber_allocations_data.append((fiber_data, wave))
                    allocations_data.append((edge_data, fiber_allocations_data))
                tickets_data[ticket.ticket_id] = allocations_data
            with open(all_tickets_path, "w") as f:
                json.dump(tickets_data, f)
        loading_end = time.time()
        print(
            f"Loading/computing {len(self.all_tickets)} tickets took {loading_end - loading_start:.4f} seconds"
        )
        assert len(self.all_tickets) > 0, "No tickets found"
        # Initialize all_klee_var_names for the demand variables
        self.all_klee_var_names = []
        # For Arrow problem, we have demands between all pairs of nodes
        # Determine number of nodes dynamically from the topology
        self.num_nodes = len(self.all_nodes)

        # Create node mapping from string names to integer indices
        node_list = sorted(self.all_nodes)
        self.node_to_index = {node: i for i, node in enumerate(node_list)}
        self.index_to_node = {i: node for i, node in enumerate(node_list)}

        for from_ in range(self.num_nodes):
            for to_ in range(self.num_nodes):
                if from_ != to_:
                    self.all_klee_var_names.append(f"demand_{from_}_{to_}")

        self.num_total_klee_inputs = len(self.all_klee_var_names)
        print(f"Number of nodes: {self.num_nodes}")
        print(f"Node mapping: {self.node_to_index}")
        print(f"Number of possible demands: {len(self.all_klee_var_names)}")
        print(f"All Klee variable names: {self.all_klee_var_names}")

        self.arrow_selected_tickets_per_scenario_per_seed = {} # scenario_str -> seed -> tickets (list of LotteryTickets for that seed used in arrow)
        for scenario_index, (all_tickets, _, scenario) in enumerate(self.all_tickets_and_nodes_for_scenarios):
            scenario_str = "fiber_cut_" + "-".join(str(fiber_id) for fiber_id in scenario)
            self.arrow_selected_tickets_per_scenario_per_seed[scenario_str] = {}
            for seed in self.seeds:
                random.seed(seed)
                self.arrow_selected_tickets_per_scenario_per_seed[scenario_str][seed] = random.sample(
                    all_tickets, min(self.num_tickets, len(all_tickets))
                )

        # limit the number of tickets to 1000 for optimal
        self.all_tickets = self.all_tickets[:self.max_num_optimal_tickets]
        # Make sure all the selected tickets are in all_tickets
        for scenario_str, seed_tickets in self.arrow_selected_tickets_per_scenario_per_seed.items():
            for seed, tickets in seed_tickets.items():
                for ticket in tickets:
                    if ticket not in self.all_tickets:
                        self.all_tickets.append(ticket)

    def _extract_demands_from_input_dict(
        self, input_dict: Dict[str, float]
    ) -> Dict[Tuple[str, str], int]:
        """
        Extract demands from input_dict.
        Looks for keys like 'demand_s_t' and converts them to (node_s, node_t) tuples.
        """
        demands = {}
        for key, value in input_dict.items():
            if key.startswith("demand_"):
                parts = key.split("_")
                if len(parts) == 3:  # demand_s_t format
                    try:
                        s_idx, t_idx = int(parts[1]), int(parts[2])
                        # Convert indices back to node names
                        if (
                            hasattr(self, "index_to_node")
                            and s_idx in self.index_to_node
                            and t_idx in self.index_to_node
                        ):
                            # s, t = self.index_to_node[s_idx], self.index_to_node[t_idx]
                            s, t = s_idx, t_idx
                            demands[(s, t)] = int(value)
                    except ValueError:
                        continue  # Skip invalid demand keys
        return demands

    def modify_ticket_for_scenario(
        self, ticket: LotteryTicket, scenario: List[int]
    ) -> LotteryTicket:
        """
        The scenario is a list of fibers that are failed. Go through the ticket and remove the fibers that are in the scenario.
        Then, return the modified ticket and the number of nodes in the modified ticket.
        """
        modified_ticket = ticket.copy()
        modified_ticket.allocations = [
            (
                edge,
                [
                    (fiber, wave)
                    for fiber, wave in fiber_allocations
                    if fiber.fiber_id not in scenario
                ],
            )
            for edge, fiber_allocations in ticket.allocations
        ]
        modified_ticket.fix_edge_capacities()
        # if edge capacity is 0, remove the edge
        modified_ticket.allocations = [
            (edge, fiber_allocations)
            for edge, fiber_allocations in modified_ticket.allocations
            if edge.capacity > 0
        ]
        return modified_ticket

    def get_all_new_tickets_and_nodes_for_scenario(
        self, fiber_dict: Dict[int, Fiber], scenario: List[int]
    ) -> Tuple[List[LotteryTicket], List[str]]:
        """
        Get all the new tickets and the number of nodes for the scenario.
        """
        new_fiber_dict = {
            fiber_id: fiber
            for fiber_id, fiber in fiber_dict.items()
            if fiber_id not in scenario
        }
        new_tickets, new_nodes, _, _ = generate_all_tickets(new_fiber_dict)
        return new_tickets, new_nodes

    def get_all_new_tickets_and_nodes_for_scenario_from_all_tickets(
        self, scenario: List[int]
    ) -> Tuple[List[LotteryTicket], List[str]]:
        """
        Get all the new tickets and the number of nodes for the scenario.
        """
        new_tickets = []
        new_nodes = set()
        for ticket in self.all_tickets:
            modified_ticket = self.modify_ticket_for_scenario(ticket, scenario)
            if len(modified_ticket.allocations) == 0:
                continue
            new_tickets.append(modified_ticket)
            for edge, fiber_allocations in modified_ticket.allocations:
                for fiber, _ in fiber_allocations:
                    for node in fiber.fiber_ip_path:
                        new_nodes.add(node)
        new_tickets = self.remove_duplicate_tickets(new_tickets)
        return new_tickets, list(new_nodes)

    def remove_duplicate_tickets(
        self, tickets: List[LotteryTicket]
    ) -> List[LotteryTicket]:
        """
        Remove duplicate tickets that represent the same logical topology.

        For tickets with the same IP topology structure, keep only the one with maximum total capacity.
        This ensures that after a fiber failure, we have the most capable version of each distinct topology.
        """
        # TODO: this is not working for larger topologies, needs to be fixed
        print(f"This is not working for larger topologies, needs to be fixed")
        if not tickets:
            return []

        # Group tickets by their topology structure
        topology_groups = {}
        for ticket in tickets:
            # Create a normalized representation of the ticket topology
            topology_key = self._normalize_ticket(ticket)

            if topology_key not in topology_groups:
                topology_groups[topology_key] = []
            topology_groups[topology_key].append(ticket)

        # For each topology group, keep only the ticket with maximum total capacity
        unique_tickets = []
        for topology_key, ticket_group in topology_groups.items():
            if len(ticket_group) == 1:
                # Only one ticket with this topology, keep it
                unique_tickets.append(ticket_group[0])
            else:
                # Multiple tickets with same topology, find the one with max capacity
                for ticket in ticket_group:
                    total_cap = self._calculate_total_capacity(ticket)

                max_capacity_ticket = max(
                    ticket_group, key=self._calculate_total_capacity
                )
                max_cap = self._calculate_total_capacity(max_capacity_ticket)
                unique_tickets.append(max_capacity_ticket)

        removed_count = len(tickets) - len(unique_tickets)
        if removed_count > 0:
            print(
                f"\nRemoved {removed_count} duplicate tickets, kept {len(unique_tickets)} unique topologies"
            )

        return unique_tickets

    def _calculate_total_capacity(self, ticket: LotteryTicket) -> float:
        """
        Calculate the total capacity of a ticket by summing all edge capacities.
        """
        total_capacity = 0.0
        for edge, _ in ticket.allocations:
            total_capacity += edge.capacity
        return total_capacity

    def _normalize_ticket(self, ticket: LotteryTicket) -> str:
        """
        Create a normalized string representation of a ticket for duplicate detection.

        The normalization should distinguish between different logical topologies:
        1. Different edge structures (different sets of edges)
        2. Different capacity ratios between edges
        3. Different fiber allocation patterns

        This ensures that tickets with genuinely different logical structures are not considered duplicates.
        """
        if not ticket.allocations:
            return "empty"

        # Extract edge information and create a more detailed signature
        edge_info = []

        for edge, fiber_allocations in ticket.allocations:
            # Sort fiber allocations by fiber_id for consistency
            sorted_fibers = sorted(fiber_allocations, key=lambda x: x[0].fiber_id)

            # Create fiber allocation string (include wavelength counts for proper distinction)
            fiber_str = ",".join(
                f"{fiber.fiber_id}:{wave}" for fiber, wave in sorted_fibers
            )

            edge_info.append(f"{edge.source}->{edge.target}:{fiber_str}")

        # Sort edges for consistent ordering
        edge_info.sort()

        return "|".join(edge_info)

    def arrow(
        self,
        demands: Dict[Tuple[str, str], int],
        selected_tickets: List[LotteryTicket],
    ):
        results = []
        ticket_ids = [ticket.ticket_id for ticket in selected_tickets]

        tickets_with_results = []

        for ticket in selected_tickets:
            ticket_converted = ticket_to_demand_pinning_format(ticket)
            ticket_result = optimal_TE_for_arrow(
                ticket_converted["num_nodes"],
                ticket_converted["edges"],
                demands,
            )
            results.append(ticket_result)
            tickets_with_results.append((ticket, ticket_converted, ticket_result))

        best_result = max(results, key=lambda x: x["optimal_total_flow"])

        # Find which ticket produced the best result
        best_ticket = None
        best_ticket_converted = None
        for ticket, ticket_converted, result in tickets_with_results:
            if result["optimal_total_flow"] == best_result["optimal_total_flow"]:
                best_ticket = ticket
                best_ticket_converted = ticket_converted
                break

        # Enhance all_vars with additional variables (similar to optimal method)
        enhanced_all_vars = best_result["all_vars"].copy()
        # Add weight variables indicating which tickets were considered (heuristic selection)
        # For arrow method, we can't determine exact indices since we randomly sample,
        # but we can indicate that it's a heuristic selection
        for k in ticket_ids:
            if k != best_ticket.ticket_id:
                enhanced_all_vars[f"weight_{k}"] = (
                    0.0  # Not explicitly selected in heuristic
                )
            else:
                enhanced_all_vars[f"weight_{k}"] = 1.0
        # Note: In a real implementation, we'd track which tickets were sampled

        # Add effective capacities from the selected ticket
        effective_capacities = {}
        if best_ticket_converted:
            for edge, capacity in best_ticket_converted["edges"].items():
                effective_capacities[edge] = capacity
                enhanced_all_vars[f"effective_capacity_{edge}"] = capacity

        # Add Arrow-specific dual variables that the filtering process expects
        # Map TE solver dual variables to Arrow format
        new_vars = {}
        for key, value in enhanced_all_vars.items():
            if key.startswith("lambda_flow_demand_"):
                # Convert lambda_flow_demand_{s}_{t} to lambda_dem_{s}_{t}
                parts = key.split("_")
                s, t = parts[3], parts[4]  # lambda_flow_demand_{s}_{t}
                new_vars[f"lambda_dem_{s}_{t}"] = value

        # Add zero values for any missing dual variables that the filtering process might expect
        for s in range(self.num_nodes):
            for t in range(self.num_nodes):
                if s != t:
                    # Add lambda_dem_{s}_{t} if not present
                    lambda_dem_key = f"lambda_dem_{s}_{t}"
                    if lambda_dem_key not in enhanced_all_vars:
                        new_vars[lambda_dem_key] = 0.0

                    # Add lambda_flow_cons_{s}_{t}_{n} for all nodes n
                    for n in range(self.num_nodes):
                        if f"lambda_flow_cons_{s}_{t}_{n}" not in enhanced_all_vars:
                            new_vars[f"lambda_flow_cons_{s}_{t}_{n}"] = 0.0

                    # Add lambda_capacity_edge_{u}_{v} for all edges
                    for u in range(self.num_nodes):
                        for v in range(self.num_nodes):
                            if u != v:
                                if (
                                    f"lambda_capacity_edge_{u}_{v}"
                                    not in enhanced_all_vars
                                ):
                                    new_vars[f"lambda_capacity_edge_{u}_{v}"] = 0.0

                    # Add flow variables f_{s}_{t}_{u}_{v} for all edges
                    for u in range(self.num_nodes):
                        for v in range(self.num_nodes):
                            if u != v:
                                flow_key = f"f_{s}_{t}_{u}_{v}"
                                if flow_key not in enhanced_all_vars:
                                    new_vars[flow_key] = 0.0

                    # Add flow_{s}_{t} variables
                    flow_total_key = f"flow_{s}_{t}"
                    if flow_total_key not in enhanced_all_vars:
                        new_vars[flow_total_key] = 0.0

        # Add lambda_weight_sum if not present
        if "lambda_weight_sum" not in enhanced_all_vars:
            new_vars["lambda_weight_sum"] = 0.0

        # Add all new variables at once
        enhanced_all_vars.update(new_vars)

        return {
            "optimal_total_flow": best_result["optimal_total_flow"],
            "flow_values": best_result["flow_values"],
            "flowpath_values": best_result["flowpath_values"],
            "all_vars": enhanced_all_vars,  # Enhanced with capacity variables
            "effective_capacities": effective_capacities,  # Capacities from selected ticket
            "heuristic": True,  # Flag to indicate this is the heuristic result
            "selected_tickets": selected_tickets,
            "best_ticket": best_ticket,
        }
   # TODO: adjust comment --- it doesnt' make the heuristic deterministic, it allows you to find average case adversarial scenarios where you take the frequentists approach to probability.
    def arrow_wrapper(self, demands: Dict[Tuple[str, str], int]):
        """
        Wrapper function that runs Arrow heuristic multiple times and averages the results.
        This makes the probabilistic Arrow heuristic deterministic.
        """
        if ENABLE_PARALLEL_PROCESSING and len(self.arrow_selected_tickets_per_scenario_per_seed) > 1:
            # Use parallel processing for scenarios
            num_cores = min(cpu_count(), len(self.arrow_selected_tickets_per_scenario_per_seed))
            if ENABLE_PRINT:
                print(f"Using parallel processing with {num_cores} cores for arrow_wrapper")

            # Prepare arguments for parallel processing
            args_list = [
                (demands, selected_tickets_per_seed, self)
                for _, selected_tickets_per_seed in self.arrow_selected_tickets_per_scenario_per_seed.items()
            ]

            # Process scenarios in parallel
            try:
                with Pool(processes=num_cores) as pool:
                    scenario_solutions = pool.map(_process_arrow_scenario, args_list)
            except Exception as e:
                if ENABLE_PRINT:
                    print(f"Parallel processing failed, falling back to sequential: {e}")
                # Fall back to sequential processing
                scenario_solutions = []
                for _, selected_tickets_per_seed in self.arrow_selected_tickets_per_scenario_per_seed.items():
                    scenario_solution_list = []
                    for _, selected_tickets in selected_tickets_per_seed.items():
                        solution = self.arrow(demands, selected_tickets=selected_tickets)
                        scenario_solution_list.append(solution)
                    scenario_solutions.append(scenario_solution_list)

            # Flatten the solutions from all scenarios
            solutions = []
            for scenario_solution_list in scenario_solutions:
                solutions.extend(scenario_solution_list)
        else:
            # Fall back to sequential processing
            solutions = []
            for _, selected_tickets_per_seed in self.arrow_selected_tickets_per_scenario_per_seed.items():
                for _, selected_tickets in selected_tickets_per_seed.items():
                    solution = self.arrow(demands, selected_tickets=selected_tickets)
                    solutions.append(solution)

        # Average the solutions
        num_random = len(solutions)
        heuristic_value = (
            sum(solution["optimal_total_flow"] for solution in solutions) / num_random
        )

        # Create a combined code path number (concatenate all results)
        code_path_num = ""
        for solution in solutions:
            best_ticket = solution["best_ticket"]
            code_path_num += f"{best_ticket.ticket_id}"

        # Average the all_vars
        all_vars_keys = set()
        for solution in solutions:
            all_vars_keys.update(solution["all_vars"].keys())
        all_vars_keys = list(all_vars_keys)
        all_vars = {
            key: sum(solution["all_vars"].get(key, 0.0) for solution in solutions)
            / num_random
            for key in all_vars_keys
        }

        # Add information about the averaging
        all_vars["arrow_num_runs"] = num_random
        all_vars["arrow_individual_results"] = [
            sol["optimal_total_flow"] for sol in solutions
        ]

        return {
            "optimal_total_flow": heuristic_value,
            "flow_values": solutions[0]["flow_values"],
            "flowpath_values": solutions[0]["flowpath_values"],
            "all_vars": all_vars,
            "effective_capacities": solutions[0]["effective_capacities"],
            "heuristic": True,
            "code_path_num": code_path_num,
        }
  # TODO: needs at the very least a comment about what is happening here in the code.
    def optimal_wrapper(self, demands: Dict[Tuple[str, str], int]):
        if ENABLE_PARALLEL_PROCESSING and len(self.all_tickets_and_nodes_for_scenarios) > 1:
            # Use parallel processing for scenarios
            num_cores = min(cpu_count(), len(self.all_tickets_and_nodes_for_scenarios))
            if ENABLE_PRINT:
                print(f"Using parallel processing with {num_cores} cores for optimal_wrapper")

            # Prepare arguments for parallel processing
            args_list = [
                (demands, all_tickets, self)
                for _, (all_tickets, _, _) in enumerate(self.all_tickets_and_nodes_for_scenarios)
            ]

            # Process scenarios in parallel
            try:
                with Pool(processes=num_cores) as pool:
                    all_solutions = pool.map(_process_optimal_scenario, args_list)
            except Exception as e:
                if ENABLE_PRINT:
                    print(f"Parallel processing failed, falling back to sequential: {e}")
                # Fall back to sequential processing
                all_solutions = []
                for scenario_idx, (all_tickets, _, _) in enumerate(
                    self.all_tickets_and_nodes_for_scenarios
                ):
                    solution = self.optimal(demands, all_tickets)
                    all_solutions.append(solution)
        else:
            # Fall back to sequential processing
            all_solutions = []
            for scenario_idx, (all_tickets, _, _) in enumerate(
                self.all_tickets_and_nodes_for_scenarios
            ):
                solution = self.optimal(demands, all_tickets)
                all_solutions.append(solution)

        # Average optimal_total_flow and all_vars
        all_vars_keys = set()
        for solution in all_solutions:
            all_vars_keys.update(solution["all_vars"].keys())
        all_vars_keys = list(all_vars_keys)
        optimal_sol = {
            "optimal_total_flow": sum(
                solution["optimal_total_flow"] for solution in all_solutions
            )
            / len(all_solutions),
            "all_vars": {
                key: sum(
                    solution["all_vars"].get(key, 0.0) for solution in all_solutions
                )
                / len(all_solutions)
                for key in all_vars_keys
            },
            "relaxed": False,
        }
        return optimal_sol
    # TODO: needs a comment, especially about the difference between this and the function at the top of the file.
    def optimal(
        self, demands: Dict[Tuple[str, str], int], all_tickets: List[LotteryTicket]
    ):
        if not all_tickets:
            raise ValueError("No tickets available — cannot compute optimal solution.")
        # Pre-convert tickets once: (ticket_id, conv, sol)
        best_tuple = None
        converted = []  # keep (ticket_id, conv, sol_or_None) so we can reuse conv

        if ENABLE_PARALLEL_PROCESSING and len(all_tickets) > 1:
            # Use parallel processing for ticket evaluation
            num_cores = min(cpu_count(), len(all_tickets))
            if ENABLE_PRINT:
                print(f"Using parallel processing with {num_cores} cores for ticket evaluation")

            # Prepare arguments for parallel processing
            args_list = [(ticket, demands) for ticket in all_tickets]

            # Process tickets in parallel
            try:
                with Pool(processes=num_cores) as pool:
                    converted = pool.map(_process_ticket_evaluation, args_list)
            except Exception as e:
                if ENABLE_PRINT:
                    print(f"Parallel processing failed, falling back to sequential: {e}")
                # Fall back to sequential processing
                converted = []
                for ticket in all_tickets:
                    ticket_id = ticket.ticket_id
                    conv = ticket_to_demand_pinning_format(ticket)
                    sol = optimal_TE_for_arrow(conv["num_nodes"], conv["edges"], demands)
                    converted.append((ticket_id, conv, sol))
        else:
            # Sequential processing
            for ticket in all_tickets:
                ticket_id = ticket.ticket_id
                conv = ticket_to_demand_pinning_format(ticket)
                sol = optimal_TE_for_arrow(conv["num_nodes"], conv["edges"], demands)
                converted.append((ticket_id, conv, sol))

        # Find the best solution
        for ticket_id, conv, sol in converted:
            # tie-breaker: keep first with strictly better objective (deterministic)
            if (
                best_tuple is None
                or sol["optimal_total_flow"] > best_tuple[2]["optimal_total_flow"]
            ):
                best_tuple = (ticket_id, conv, sol)

        ticket_id, conv, sol = best_tuple
        all_vars: dict = {}

        # Copy primal/path/weights and any duals already exposed by the solver
        _copy_vars(
            all_vars,
            sol["flow_values"],
            sol["flowpath_values"],
            weight_vals={
                f"weight_{i}": (1.0 if i == ticket_id else 0.0)
                for i in [ticket.ticket_id for ticket in all_tickets]
            },
            duals=sol["all_vars"].get("dual_values", {}),
        )

        # Union of ALL edges across tickets (reuse cached conversions)
        all_edges = {e for _, c, _ in converted for e in c["edges"]}

        # # Effective capacities for every edge seen in any ticket, under the chosen ticket
        # for e in all_edges:
        #     all_vars[f"effective_capacity_{e}"] = conv["edges"].get(e, 0.0)

        # Add demand_* inputs so both sides align
        for (s, t), dem in demands.items():
            all_vars[f"demand_{s}_{t}"] = float(dem)

        # Align with relaxed: lambda_weight_sum exists (no weight constraint in optimal)
        all_vars["lambda_weight_sum"] = 0.0

        # If your per-ticket solver names capacity duals differently, map them here.
        # Example: move "capacity_edge_0_1" -> "lambda_capacity_edge_0_1"
        duals = sol["all_vars"].get("dual_values", {})
        for e in all_edges:
            relaxed_key = f"lambda_capacity_{e}"  # "lambda_capacity_edge_u_v"
            alt_key = f"capacity_{e}"  # e.g., "capacity_edge_u_v"
            if relaxed_key not in all_vars:
                if alt_key in duals:
                    all_vars[relaxed_key] = float(duals[alt_key])
                else:
                    all_vars[relaxed_key] = 0.0  # placeholder to match relaxed

        # Add zero placeholders for all possible paths to match relaxed methods
        # This ensures consistency in variable naming between optimal and relaxed
        num_nodes = conv["num_nodes"]  # Get num_nodes from the selected ticket
        for s, t in demands:
            try:
                # Create problem_config with edges for find_all_paths
                problem_config_with_edges = {"num_nodes": num_nodes}
                for edge in all_edges:
                    problem_config_with_edges[edge] = 1.0  # Default capacity

                cand_paths = find_all_paths(
                    problem_config_with_edges, str(s), str(t), max_num_paths=10
                )
                for p in cand_paths:
                    key = "path_flow_" + "_".join(
                        p
                    )  # p is list of strings from find_all_paths
                    if key not in all_vars:
                        all_vars[key] = 0.0  # Zero placeholder for unused paths
            except NameError:
                # If find_all_paths isn't available here, skip padding
                print("find_all_paths not available, skipping padding")
                pass

        # Define helper function for parsing edge strings
        def parse_edge(e_str):
            _, u, v = e_str.split("_")
            return (int(u), int(v))

        # Add zero dual variables for flow conservation constraints that don't exist
        # This ensures all_vars has entries for any flow conservation constraint the gradient might reference
        for s in range(num_nodes):
            for t in range(num_nodes):
                if s != t:  # All possible demand pairs
                    for n in range(num_nodes):  # All nodes
                        key = f"lambda_flow_cons_{s}_{t}_{n}"
                        if key not in all_vars:
                            all_vars[key] = 0.0

                    # Also ensure demand dual variables exist for all possible pairs
                    dem_key = f"lambda_dem_{s}_{t}"
                    if dem_key not in all_vars:
                        all_vars[dem_key] = 0.0

                    # Add zero demand entries for all possible demand pairs not in the current demands
                    demand_key = f"demand_{s}_{t}"
                    if demand_key not in all_vars:
                        all_vars[demand_key] = 0.0

                    # Add zero edge flow variables for all possible demand pairs and edges
                    for edge_str in all_edges:
                        u, v = parse_edge(edge_str)
                        edge_flow_key = f"f_{s}_{t}_{u}_{v}"
                        if edge_flow_key not in all_vars:
                            all_vars[edge_flow_key] = 0.0

        return {
            "optimal_total_flow": sol["optimal_total_flow"],
            "all_vars": all_vars,
            "selected_ticket_idx": ticket_id,
            "relaxed": False,
        }
   # TODO: this function needs a comment.
    def relaxed_optimal_wrapper(self, demands: Dict[Tuple[str, str], int]):
        if ENABLE_PARALLEL_PROCESSING and len(self.all_tickets_and_nodes_for_scenarios) > 1:
            # Use parallel processing for scenarios
            num_cores = min(cpu_count(), len(self.all_tickets_and_nodes_for_scenarios))
            if ENABLE_PRINT:
                print(f"Using parallel processing with {num_cores} cores for relaxed_optimal_wrapper")

            # Prepare arguments for parallel processing
            args_list = [
                (demands, all_tickets, scenario, self)
                for scenario_idx, (all_tickets, _, scenario) in enumerate(self.all_tickets_and_nodes_for_scenarios)
            ]

            # Process scenarios in parallel
            try:
                with Pool(processes=num_cores) as pool:
                    all_solutions = pool.map(_process_relaxed_optimal_scenario, args_list)
            except Exception as e:
                if ENABLE_PRINT:
                    print(f"Parallel processing failed, falling back to sequential: {e}")
                # Fall back to sequential processing
                all_solutions = []
                for scenario_idx, (all_tickets, _, scenario) in enumerate(
                    self.all_tickets_and_nodes_for_scenarios
                ):
                    try:
                        solution = self.relaxed_optimal(demands, all_tickets)
                        all_solutions.append(solution)
                    except Exception as e:
                        print(
                            f"Error in relaxed_optimal for scenario {scenario_idx}: {scenario}"
                        )
                        print(f"All tickets: {all_tickets}")
                        raise e
        else:
            # Fall back to sequential processing
            all_solutions = []
            for scenario_idx, (all_tickets, _, scenario) in enumerate(
                self.all_tickets_and_nodes_for_scenarios
            ):
                try:
                    solution = self.relaxed_optimal(demands, all_tickets)
                    all_solutions.append(solution)
                except Exception as e:
                    print(
                        f"Error in relaxed_optimal for scenario {scenario_idx}: {scenario}"
                    )
                    print(f"All tickets: {all_tickets}")
                    raise e

        # Average optimal_total_flow and all_vars
        all_vars_keys = set()
        for solution in all_solutions:
            all_vars_keys.update(solution["all_vars"].keys())
        all_vars_keys = list(all_vars_keys)
        optimal_sol = {
            "optimal_total_flow": sum(
                solution["optimal_total_flow"] for solution in all_solutions
            )
            / len(all_solutions),
            "all_vars": {
                key: sum(
                    solution["all_vars"].get(key, 0.0) for solution in all_solutions
                )
                / len(all_solutions)
                for key in all_vars_keys
            },
            "relaxed": True,
        }
        return optimal_sol

    def relaxed_optimal(
        self,
        demands: Dict[Tuple[str, str], int],
        all_tickets: List[LotteryTicket],
        add_zero_placeholders: bool = True,
        max_num_paths: int = 10,
    ):
        """
        Cross-layer relaxed LP (edge form)

        max   sum_{s,t} F_{s,t}
        s.t.  ∀(s,t),∀n:   ∑_{(u,v): v=n} f_{s,t}^{u,v} - ∑_{(u,v): u=n} f_{s,t}^{u,v}
                            = -F_{s,t}·1[n=s] + F_{s,t}·1[n=t]
            ∀(u,v):      ∑_i w_i cap_i^{u,v} - ∑_{s,t} f_{s,t}^{u,v}  ≥ 0
            ∀(s,t):      d_{s,t} - F_{s,t} ≥ 0
                            ∑_i w_i = 1
            F,f,w ≥ 0
        """
        from ortools.linear_solver import pywraplp

        solver = pywraplp.Solver.CreateSolver("GLOP")
        if not solver:
            raise Exception("Solver could not be created!")

        # ---- Tickets → per-edge capacities, and union of all edges ----
        ticket_caps: dict[str, dict] = {}
        all_edges = set()
        for t in all_tickets:
            conv = ticket_to_demand_pinning_format(t)
            ticket_caps[f"weight_{t.ticket_id}"] = conv["edges"]  # { "edge_u_v": cap }
            all_edges |= set(conv["edges"].keys())
        if not ticket_caps:
            raise ValueError("No valid tickets found")

        num_nodes = self.num_nodes

        def parse_edge(e_str: str) -> tuple[int, int]:
            # "edge_0_3" -> (0, 3)
            _, u, v = e_str.split("_")
            return int(u), int(v)

        # Use a deterministic edge order (fixes set/zip ordering bugs)
        edge_strs = sorted(all_edges)
        edge_list = [parse_edge(e) for e in edge_strs]  # aligns 1:1 with edge_strs
        ticket_ids = [ticket.ticket_id for ticket in all_tickets]
        # ---- Variables ----
        weight = {
            f"weight_{k}": solver.NumVar(0.0, solver.infinity(), f"weight_{k}")
            for k in ticket_ids
        }
        # Create variables for ALL possible demand pairs (not just the ones with non-zero demands)
        all_demand_pairs = [
            (s, t) for s in range(num_nodes) for t in range(num_nodes) if s != t
        ]

        # IMPORTANT: F is unbounded above (objective is max) and capped by explicit demand constraints
        F = {
            (s, t): solver.NumVar(0.0, solver.infinity(), f"flow_{s}_{t}")
            for (s, t) in all_demand_pairs
        }
        f = {
            (s, t, u, v): solver.NumVar(0.0, solver.infinity(), f"f_{s}_{t}_{u}_{v}")
            for (s, t) in all_demand_pairs
            for (u, v) in edge_list
        }

        # ---- Constraints ----
        # Sum of weights = 1
        wsum_ct = solver.Add(solver.Sum(weight[k] for k in weight) == 1.0)

        # Demand caps: F_{s,t} ≤ d_{s,t} (only for demand pairs that have non-zero demands)
        dem_cap_ct = {}
        for (s, t), dem in demands.items():
            dem_cap_ct[(s, t)] = solver.Add(F[(s, t)] <= float(dem))

        # For demand pairs not in demands, constraint F_{s,t} ≤ 0 (no flow allowed)
        for s, t in all_demand_pairs:
            if (s, t) not in demands:
                dem_cap_ct[(s, t)] = solver.Add(F[(s, t)] <= 0.0)

        # Flow conservation: inflow - outflow = {-F, 0, +F}
        flow_cons_ct = {}
        for s, t in all_demand_pairs:
            for n in range(num_nodes):
                inflow = solver.Sum(f[(s, t, u, v)] for (u, v) in edge_list if v == n)
                outflow = solver.Sum(f[(s, t, u, v)] for (u, v) in edge_list if u == n)
                rhs = -F[(s, t)] if n == s else (F[(s, t)] if n == t else 0.0)
                flow_cons_ct[(s, t, n)] = solver.Add(inflow - outflow == rhs)

        # Edge capacities:  ∑_i w_i cap_i(e) - ∑_{s,t} f_{s,t}(e) ≥ 0
        cap_ct = {}
        for (u, v), e_str in zip(edge_list, edge_strs):
            rhs_cap = solver.Sum(
                weight[f"weight_{k}"] * ticket_caps[f"weight_{k}"].get(e_str, 0.0)
                for k in ticket_ids
            )
            lhs_flow = solver.Sum(f[(s, t, u, v)] for (s, t) in all_demand_pairs)
            cap_ct[(u, v)] = solver.Add(lhs_flow <= rhs_cap)

        # ---- Objective ----
        solver.Maximize(solver.Sum(F[(s, t)] for (s, t) in demands))

        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            raise Exception("Relaxed problem did not solve to optimality")

        # ---- Collect primal values ----
        optimal_weights = {k: weight[k].solution_value() for k in weight}
        flow_values = {
            f"flow_{s}_{t}": F[(s, t)].solution_value() for (s, t) in all_demand_pairs
        }
        fval = {
            (s, t, u, v): f[(s, t, u, v)].solution_value()
            for (s, t) in all_demand_pairs
            for (u, v) in edge_list
        }

        # ---- Path-flow reconstruction (flow decomposition) ----
        EPS = 1e-9
        path_flow_values: dict[str, float] = {}

        def decompose_paths(s: int, t: int):
            # residual per-edge for this commodity
            resid = {
                (u, v): fval[(s, t, u, v)]
                for (u, v) in edge_list
                if fval[(s, t, u, v)] > EPS
            }
            adj: dict[int, list[tuple[int, float]]] = {}
            for (u, v), val in resid.items():
                adj.setdefault(u, []).append((v, val))

            def find_path():
                # DFS on positive residuals
                stack = [(s, [s])]
                seen = set()
                while stack:
                    node, path = stack.pop()
                    if node == t:
                        return path
                    if node in seen:
                        continue
                    seen.add(node)
                    for nv, val in adj.get(node, []):
                        if val > EPS and nv not in path:  # simple path
                            stack.append((nv, path + [nv]))
                return None

            while True:
                p = find_path()
                if p is None:
                    break
                bott = min(resid[(p[i], p[i + 1])] for i in range(len(p) - 1))
                for i in range(len(p) - 1):
                    u, v = p[i], p[i + 1]
                    resid[(u, v)] -= bott
                    if resid[(u, v)] <= EPS:
                        resid[(u, v)] = 0.0
                adj.clear()
                for (u, v), val in resid.items():
                    if val > EPS:
                        adj.setdefault(u, []).append((v, val))
                key = "path_flow_" + "_".join(str(x) for x in p)
                path_flow_values[key] = path_flow_values.get(key, 0.0) + bott

        for s, t in demands:
            if flow_values[f"flow_{s}_{t}"] > EPS:
                decompose_paths(s, t)

        # Optional: add 0-placeholders for uniform keys across methods
        if add_zero_placeholders:
            try:
                problem_config_with_edges = {"num_nodes": num_nodes}
                for e in edge_strs:
                    problem_config_with_edges[e] = 1.0
                for s, t in demands:
                    cand_paths = find_all_paths(
                        problem_config_with_edges,
                        str(s),
                        str(t),
                        max_num_paths=max_num_paths,
                    )
                    for p in cand_paths:
                        key = "path_flow_" + "_".join(p)
                        if key not in path_flow_values:
                            path_flow_values[key] = 0.0
            except NameError:
                pass

        # ---- Effective capacities under the convex combination of tickets ----
        effective_capacities = {
            e_str: sum(
                optimal_weights[f"weight_{k}"]
                * ticket_caps[f"weight_{k}"].get(e_str, 0.0)
                for k in ticket_ids
            )
            for e_str in edge_strs
        }
        # ---- Build all_vars with primals and duals ----
        all_vars: dict[str, float] = {}

        # weights
        for k, w in optimal_weights.items():
            all_vars[k] = w

        # F totals
        all_vars.update(flow_values)

        # edge flows
        for s, t in demands:
            for u, v in edge_list:
                all_vars[f"f_{s}_{t}_{u}_{v}"] = fval[(s, t, u, v)]

        # path flows
        for k, v in path_flow_values.items():
            all_vars[k] = v

        # duals
        all_vars["lambda_weight_sum"] = wsum_ct.dual_value()
        for (u, v), ct in cap_ct.items():
            all_vars[f"lambda_capacity_edge_{u}_{v}"] = ct.dual_value()
        for (s, t, n), ct in flow_cons_ct.items():
            all_vars[f"lambda_flow_cons_{s}_{t}_{n}"] = ct.dual_value()
        for (s, t), ct in dem_cap_ct.items():
            all_vars[f"lambda_dem_{s}_{t}"] = ct.dual_value()

        # Add zero dual variables for flow conservation constraints that don't exist
        # This ensures all_vars has entries for any flow conservation constraint the gradient might reference
        for s in range(num_nodes):
            for t in range(num_nodes):
                if s != t:  # All possible demand pairs
                    for n in range(num_nodes):  # All nodes
                        key = f"lambda_flow_cons_{s}_{t}_{n}"
                        if key not in all_vars:
                            all_vars[key] = 0.0

                    # Also ensure demand dual variables exist for all possible pairs
                    dem_key = f"lambda_dem_{s}_{t}"
                    if dem_key not in all_vars:
                        all_vars[dem_key] = 0.0

        # also store the demands as inputs (handy for L/∇L evals)
        for (s, t), dem in demands.items():
            all_vars[f"demand_{s}_{t}"] = float(dem)

        # Add zero demand entries for all possible demand pairs not in the current demands
        for s in range(num_nodes):
            for t in range(num_nodes):
                if s != t:
                    demand_key = f"demand_{s}_{t}"
                    if demand_key not in all_vars:
                        all_vars[demand_key] = 0.0

                    # Add zero edge flow variables for all possible demand pairs and edges
                    for u, v in edge_list:
                        edge_flow_key = f"f_{s}_{t}_{u}_{v}"
                        if edge_flow_key not in all_vars:
                            all_vars[edge_flow_key] = 0.0

        optimal_total_flow = sum(F[(s, t)].solution_value() for (s, t) in demands)

        return {
            "optimal_total_flow": optimal_total_flow,
            "flow_values": flow_values,
            "path_flow_values": path_flow_values,
            "optimal_weights": optimal_weights,
            "effective_capacities": effective_capacities,
            "all_vars": all_vars,
            "relaxed": True,
            "num_tickets_used": sum(1 for w in optimal_weights.values() if w > 1e-6),
        }

    def get_arrow_lagrangian(
        self,
        input_dict: Dict[str, float],
        all_tickets: List[LotteryTicket],
        give_relaxed_gap: bool = False,
    ):
        """
        L = Σ_{s,t} F_{s,t}
            + λ_weight_sum * (1 - Σ_i w_i)
            + Σ_{s,t,n} λ_flow_{s,t,n} * (in - out - rhs_{s,t,n})
            + Σ_{u,v} λ_cap_{u,v} * ( Σ_i w_i C_i(u,v) - Σ_{s,t} f_{s,t,u,v} )
            + Σ_{s,t} λ_dem_{s,t} * ( D_{s,t} - F_{s,t} )
        """
        start_time = time.time()
        demands = self._extract_demands_from_input_dict(input_dict)
        ticket_ids = [ticket.ticket_id for ticket in all_tickets]
        weight_keys = [f"weight_{k}" for k in ticket_ids]
        # tickets & edges
        ticket_caps = []
        all_edges = set()
        for t in all_tickets:
            conv = ticket_to_demand_pinning_format(t)
            ticket_caps.append(conv["edges"])
            all_edges |= set(conv["edges"].keys())

        num_nodes = ticket_to_demand_pinning_format(self.all_tickets[0])["num_nodes"]

        def parse_edge(e_str):
            _, u, v = e_str.split("_")
            return (int(u), int(v))

        # deterministic pairing of (u,v) with edge string
        edge_strs = sorted(all_edges)
        edge_list = [parse_edge(e) for e in edge_strs]

        lagrange = 0.0
        constraints: Dict[str, float] = {}

        # objective: sum F_{s,t}
        for s, t in demands:
            lagrange += float(input_dict[f"flow_{s}_{t}"])

        if give_relaxed_gap:
            return {"lagrange": lagrange, "constraints": constraints}

        # weight-sum: 1 - Σ w_i
        wsum = sum(input_dict[k] for k in weight_keys)
        ws_res = 1.0 - wsum
        constraints["lambda_weight_sum"] = ws_res
        lagrange += input_dict["lambda_weight_sum"] * ws_res

        # flow conservation
        for s, t in demands:
            for n in range(num_nodes):
                inflow = 0.0
                outflow = 0.0
                for u, v in edge_list:
                    val = input_dict[f"f_{s}_{t}_{u}_{v}"]
                    if v == n:
                        inflow += val
                    if u == n:
                        outflow += val
                # rhs: -F at source, +F at sink, 0 otherwise
                Fst = input_dict[f"flow_{s}_{t}"]
                rhs = (-Fst if n == s else 0.0) + (Fst if n == t else 0.0)
                res = inflow - outflow - rhs
                k = f"lambda_flow_cons_{s}_{t}_{n}"
                constraints[k] = res
                lagrange += input_dict[k] * res

        # demand caps: F_{s,t} <= D_{s,t}
        for s, t in demands:
            Fst = input_dict[f"flow_{s}_{t}"]
            # IMPORTANT: read D from input_dict if present (so numeric FD matches)
            Dst = float(input_dict[f"demand_{s}_{t}"])
            res = Dst - Fst
            k = f"lambda_dem_{s}_{t}"
            constraints[k] = res
            lagrange += input_dict[k] * res

        # capacity: Σ_i w_i C_i(u,v) - Σ_{s,t} f_{s,t,u,v} >= 0
        for (u, v), e_str in zip(edge_list, edge_strs):
            total_flow = 0.0
            for s, t in demands:
                total_flow += input_dict[f"f_{s}_{t}_{u}_{v}"]
            cap = 0.0
            for k in weight_keys:
                cap += input_dict[k] * ticket_caps[k].get(e_str, 0.0)
            res = cap - total_flow
            k = f"lambda_capacity_edge_{u}_{v}"
            constraints[k] = res
            lagrange += input_dict[k] * res

        end_time = time.time()
        if ENABLE_PRINT:
            print(f"get_arrow_lagrangian took {end_time - start_time:.4f} seconds")
        return {"lagrange": lagrange, "constraints": constraints}

    def get_arrow_lagrangian_gradient(
        self,
        input_dict: Dict[str, float],
        all_tickets: List[LotteryTicket],
    ):
        """
        ∂L/∂w_i = -λ_ws + Σ_{u,v} λ_cap_{uv} C_i(u,v)
        ∂L/∂F_{st} = 1 + λ_flow_cons_{st,s} - λ_flow_cons_{st,t} - λ_dem_{st}
        ∂L/∂f_{stuv} = λ_flow_cons_{st,v} - λ_flow_cons_{st,u} - λ_cap_{uv}
        ∂L/∂λ = constraint residual
        ∂L/∂D_{st} = λ_dem_{st}    (we expose this via the key 'demand_s_t')
        """
        start_time = time.time()
        demands = self._extract_demands_from_input_dict(input_dict)
        ticket_ids = [ticket.ticket_id for ticket in all_tickets]
        weight_keys = [f"weight_{k}" for k in ticket_ids]
        # capacities & edges
        ticket_caps = {}
        for t in all_tickets:
            conv = ticket_to_demand_pinning_format(t)
            ticket_caps[f"weight_{t.ticket_id}"] = conv["edges"]
            # Ensure all_edges includes edges from all tickets that *could* be selected
            # Reconstruct all_edges from self.all_tickets (original, full set)
        
        all_edges = set()
        for ticket in self.all_tickets: # Use self.all_tickets to get all possible edges
            conv = ticket_to_demand_pinning_format(ticket)
            all_edges.update(conv["edges"].keys())

        num_nodes = self.num_nodes # Use num_nodes from self

        def parse_edge(e_str):
            _, u, v = e_str.split("_")
            return (int(u), int(v))

        edge_strs = sorted(all_edges)
        edge_list = [parse_edge(e) for e in edge_strs]

        grad: Dict[str, float] = {}

        # pre-helpers
        def lam_flow(s, t, n):
            key = f"lambda_flow_cons_{s}_{t}_{n}"
            # Return 0.0 if key not in input_dict (to handle missing duals gracefully)
            return input_dict.get(key, 0.0) 

        # compute
        for key in input_dict:
            if key.startswith("demand_"):
                # D_{s,t} appears only in λ_dem_{s,t}(D_{s,t}-F_{s,t})
                _, s, t = key.split("_")
                s, t = int(s), int(t)
                lambda_key = f"lambda_dem_{s}_{t}"
                grad[key] = input_dict.get(lambda_key, 0.0)

            elif key in weight_keys:
                lambda_weight_sum_key = "lambda_weight_sum"
                g = -input_dict.get(lambda_weight_sum_key, 0.0) # Corrected: use get with 0.0 default
                
                for (u, v), e_str in zip(edge_list, edge_strs):
                    lambda_cap_key = f"lambda_capacity_edge_{u}_{v}"
                    # Check if the weight_key exists in ticket_caps and e_str within that ticket's edges
                    if key in ticket_caps and e_str in ticket_caps[key]:
                        g += input_dict.get(lambda_cap_key, 0.0) * ticket_caps[key].get(e_str, 0.0)
                grad[key] = g

            elif key.startswith("flow_"):
                _, s, t = key.split("_")
                s, t = int(s), int(t)
                g = 1.0
                # Corrected signs for flow conservation duals:
                g += lam_flow(s, t, s)  # +λ_source (coefficient of F is +1 at source in conservation eq)
                g -= lam_flow(s, t, t)  # -λ_sink (coefficient of F is -1 at sink in conservation eq)
                
                lambda_dem_key = f"lambda_dem_{s}_{t}"
                g -= input_dict.get(lambda_dem_key, 0.0) # Corrected: use get with 0.0 default
                grad[key] = g

            elif key.startswith("f_"):
                _, s, t, u, v = key.split("_")
                s, t, u, v = int(s), int(t), int(u), int(v)
                g = 0.0
                # Corrected signs for flow conservation duals:
                g += lam_flow(s, t, v)  # inflow at v (f is positive into v)
                g -= lam_flow(s, t, u)  # outflow at u (f is negative out of u)
                
                lambda_cap_key = f"lambda_capacity_edge_{u}_{v}"
                g -= input_dict.get(lambda_cap_key, 0.0) # Corrected: use get with 0.0 default
                grad[key] = g

            elif key == "lambda_weight_sum":
                # Ensure we sum over *all* possible weight keys, even if not in input_dict (value would be 0.0)
                wsum = sum(input_dict.get(k, 0.0) for k in [f"weight_{ticket_id}" for ticket_id in ticket_ids])
                grad[key] = 1.0 - wsum

            elif key.startswith("lambda_flow_cons_"):
                _, _, _, s, t, n = key.split("_") # Adjusted split based on updated key format if necessary
                s, t, n = int(s), int(t), int(n)
                inflow = 0.0
                outflow = 0.0
                # Iterate over all possible edges to compute inflow/outflow for a given (s,t,n)
                for u_edge, v_edge in edge_list: 
                    # Use input_dict.get(..., 0.0) for flow values
                    val = input_dict.get(f"f_{s}_{t}_{u_edge}_{v_edge}", 0.0)
                    if v_edge == n:
                        inflow += val
                    if u_edge == n:
                        outflow += val
                Fst = input_dict.get(f"flow_{s}_{t}", 0.0) # Use get with default 0.0
                rhs = (-Fst if n == s else 0.0) + (Fst if n == t else 0.0)
                grad[key] = inflow - outflow - rhs

            elif key.startswith("lambda_capacity_edge_"):
                _, _, _, u, v = key.split("_") # Adjusted split based on updated key format if necessary
                u, v = int(u), int(v)
                total_flow = 0.0
                # Iterate over all possible demand pairs to sum flow on this edge
                for s_dem, t_dem in demands: # Using demands to limit flow summation
                    flow_key = f"f_{s_dem}_{t_dem}_{u}_{v}"
                    total_flow += input_dict.get(flow_key, 0.0) # Use get with default 0.0
                e_str = f"edge_{u}_{v}"
                cap = 0.0
                for weight_k in weight_keys:
                    # Access ticket_caps via weight_k string as key
                    # Ensure ticket_caps[weight_k] exists before calling .get(e_str)
                    ticket_cap_val = ticket_caps.get(weight_k, {}).get(e_str, 0.0)
                    cap += input_dict.get(weight_k, 0.0) * ticket_cap_val
                grad[key] = cap - total_flow

            elif key.startswith("lambda_dem_"):
                _, _, s, t = key.split("_")
                s, t = int(s), int(t)
                flow_key = f"flow_{s}_{t}"
                Fst = input_dict.get(flow_key, 0.0) # Use get with default 0.0
                
                # Demand lookup should use the numerical indices directly now
                demand_key = f"demand_{s}_{t}"
                Dst = float(input_dict.get(demand_key, 0.0)) # Use get with default 0.0
                
                grad[key] = Dst - Fst
            else:
                # If a variable is not explicitly handled, assume its gradient is 0
                # This helps in robust handling of new or unexpected variables
                grad[key] = 0.0


        # Finally, negate gradients for dual variables
        for key in grad:
            if key.startswith("lambda_"):
                grad[key] = -grad[key]
        # print(f"get_arrow_lagrangian_gradient took {time.time() - start_time:.4f}s")
        return grad

    # TODO: there is two of these functions, what is the difference between the two, if you need both add comments to differentiate, if one is the one you ended up using remove the other one.
    def get_arrow_lagrangian_gradient2(
        self,
        input_dict: Dict[str, float],
        all_tickets: List[LotteryTicket],
    ):
        """
        ∂L/∂w_i = -λ_ws + Σ_{u,v} λ_cap_{uv} C_i(u,v)
        ∂L/∂F_{st} = 1 + λ_flow_{st,s} - λ_flow_{st,t} - λ_dem_{st}
        ∂L/∂f_{stuv} = λ_flow_{st,v} - λ_flow_{st,u} - λ_cap_{uv}
        ∂L/∂λ = constraint residual
        ∂L/∂D_{st} = λ_dem_{st}    (we expose this via the key 'demand_s_t')
        """
        start_time = time.time()
        demands = self._extract_demands_from_input_dict(input_dict)
        ticket_ids = [ticket.ticket_id for ticket in all_tickets]
        weight_keys = [f"weight_{k}" for k in ticket_ids]
        # capacities & edges
        ticket_caps = {}
        all_edges = set()
        for t in all_tickets:
            conv = ticket_to_demand_pinning_format(t)
            ticket_caps[f"weight_{t.ticket_id}"] = conv["edges"]
            all_edges |= set(conv["edges"].keys())

        num_nodes = ticket_to_demand_pinning_format(all_tickets[0])["num_nodes"]

        def parse_edge(e_str):
            _, u, v = e_str.split("_")
            return (int(u), int(v))

        edge_strs = sorted(all_edges)
        edge_list = [parse_edge(e) for e in edge_strs]

        grad: Dict[str, float] = {}

        # pre-helpers
        def lam_flow(s, t, n):
            key = f"lambda_flow_cons_{s}_{t}_{n}"
            if key not in input_dict:
                return 0.0
            return input_dict[key]

        # compute
        for key in input_dict:
            if key.startswith("demand_"):
                # D_{s,t} appears only in λ_dem_{s,t}(D_{s,t}-F_{s,t})
                _, s, t = key.split("_")
                s, t = int(s), int(t)
                lambda_key = f"lambda_dem_{s}_{t}"
                if lambda_key not in input_dict:
                    grad[key] = 0.0
                else:
                    grad[key] = input_dict[lambda_key]

            elif key in weight_keys:
                lambda_weight_sum_key = "lambda_weight_sum"
                if lambda_weight_sum_key not in input_dict:
                    g = 0.0
                else:
                    g = -input_dict[lambda_weight_sum_key]
                for (u, v), e_str in zip(edge_list, edge_strs):
                    lambda_cap_key = f"lambda_capacity_edge_{u}_{v}"
                    if lambda_cap_key not in input_dict:
                        continue
                    # if i >= len(ticket_caps):
                    #     continue
                    g += input_dict.get(lambda_cap_key, 0.0) * ticket_caps[key].get(
                        e_str, 0.0
                    )
                grad[key] = g

            elif key.startswith("flow_"):
                _, s, t = key.split("_")
                s, t = int(s), int(t)
                g = 1.0
                g += lam_flow(s, t, s)  # +λ_source
                g -= lam_flow(s, t, t)  # -λ_sink
                lambda_dem_key = f"lambda_dem_{s}_{t}"
                if lambda_dem_key not in input_dict:
                    g -= 0.0
                else:
                    g -= input_dict[lambda_dem_key]
                grad[key] = g

            elif key.startswith("f_"):
                _, s, t, u, v = key.split("_")
                s, t, u, v = int(s), int(t), int(u), int(v)
                g = 0.0
                g += lam_flow(s, t, v)  # inflow at v
                g -= lam_flow(s, t, u)  # outflow at u
                lambda_cap_key = f"lambda_capacity_edge_{u}_{v}"
                if lambda_cap_key not in input_dict:
                    g -= 0.0
                else:
                    g -= input_dict[lambda_cap_key]
                grad[key] = g

            elif key == "lambda_weight_sum":
                wsum = sum(input_dict[k] for k in weight_keys)
                grad[key] = 1.0 - wsum

            elif key.startswith("lambda_flow_cons_"):
                _, _, _, s, t, n = key.split("_")
                s, t, n = int(s), int(t), int(n)
                inflow = 0.0
                outflow = 0.0
                for u, v in edge_list:
                    val = input_dict[f"f_{s}_{t}_{u}_{v}"]
                    if v == n:
                        inflow += val
                    if u == n:
                        outflow += val
                Fst = input_dict[f"flow_{s}_{t}"]
                rhs = (-Fst if n == s else 0.0) + (Fst if n == t else 0.0)
                grad[key] = inflow - outflow - rhs

            elif key.startswith("lambda_capacity_edge_"):
                _, _, _, u, v = key.split("_")
                u, v = int(u), int(v)
                total_flow = 0.0
                for s, t in demands:
                    flow_key = f"f_{s}_{t}_{u}_{v}"
                    if flow_key not in input_dict:
                        continue
                    total_flow += input_dict[flow_key]
                e_str = f"edge_{u}_{v}"
                cap = 0.0
                for i in ticket_ids:
                    weight_key = f"weight_{i}"
                    if weight_key not in input_dict:
                        continue
                    cap += input_dict[weight_key] * ticket_caps[weight_key].get(
                        e_str, 0.0
                    )
                grad[key] = cap - total_flow

            elif key.startswith("lambda_dem_"):
                _, _, s, t = key.split("_")
                s, t = int(s), int(t)
                flow_key = f"flow_{s}_{t}"
                if flow_key not in input_dict:
                    Fst = 0.0
                else:
                    Fst = input_dict[flow_key]
                # Convert indices back to node names for demands lookup
                if (
                    hasattr(self, "index_to_node")
                    and s in self.index_to_node
                    and t in self.index_to_node
                ):
                    node_s, node_t = self.index_to_node[s], self.index_to_node[t]
                    demand_key = f"demand_{s}_{t}"
                    if demand_key not in input_dict:
                        Dst = 0.0
                    else:
                        Dst = float(input_dict[demand_key])
                else:
                    demand_key = f"demand_{s}_{t}"
                    if demand_key not in input_dict:
                        Dst = 0.0
                    else:
                        Dst = float(input_dict[demand_key])
                grad[key] = Dst - Fst

        for key in grad:
            if key.startswith("lambda_"):
                grad[key] = -grad[key]
        # print(f"get_arrow_lagrangian_gradient took {time.time() - start_time:.4f}s")
        return grad

    def get_arrow_lagrangian_gradient_wrapper(
        self,
        input_dict: Dict[str, float],
    ):
        """
        Wrapper function that can choose between optimized and original gradient implementations.
        """
        gradients = []
        for all_tickets, _, _ in self.all_tickets_and_nodes_for_scenarios:
            gradients.append(
                self.get_arrow_lagrangian_gradient(input_dict, all_tickets)
            )
        all_gradient_keys = set()
        for gradient in gradients:
            all_gradient_keys.update(gradient.keys())
        all_gradient_keys = list(all_gradient_keys)
        return {
            key: sum(gradient.get(key, 0.0) for gradient in gradients) / len(gradients)
            for key in all_gradient_keys
        }

    def convert_input_dict_to_args(self, input_dict):
        """
        Convert input_dict to the format expected by Arrow problem methods.
        Now that functions extract demands from input_dict internally, this is simplified.
        """
        return {"input_dict": input_dict}

    def compute_optimal_value(self, args_dict):
        """
        Compute the optimal value for the Arrow problem.
        """
        self.num_compute_optimal_value_called += 1
        input_dict = args_dict["input_dict"]
        demands = self._extract_demands_from_input_dict(input_dict)
        optimal_start = time.time()
        optimal_sol = self.optimal_wrapper(demands)
        optimal_end = time.time()
        if ENABLE_PRINT:
            print(f"optimal call took {optimal_end - optimal_start:.4f} seconds")

        # Ensure all required variables are present in optimal_sol["all_vars"]
        enhanced_all_vars = optimal_sol["all_vars"].copy()
        # Add missing variables that the gradient function expects
        new_vars = {}

        # Add missing dual variables that the filtering process expects
        for s in range(self.num_nodes):
            for t in range(self.num_nodes):
                if s != t:
                    # Add lambda_dem_{s}_{t} if not present
                    lambda_dem_key = f"lambda_dem_{s}_{t}"
                    if lambda_dem_key not in enhanced_all_vars:
                        new_vars[lambda_dem_key] = 0.0

                    # Add lambda_flow_cons_{s}_{t}_{n} for all nodes n
                    for n in range(self.num_nodes):
                        if f"lambda_flow_cons_{s}_{t}_{n}" not in enhanced_all_vars:
                            new_vars[f"lambda_flow_cons_{s}_{t}_{n}"] = 0.0

                    # Add lambda_capacity_edge_{u}_{v} for all edges
                    for u in range(self.num_nodes):
                        for v in range(self.num_nodes):
                            if u != v:
                                if (
                                    f"lambda_capacity_edge_{u}_{v}"
                                    not in enhanced_all_vars
                                ):
                                    new_vars[f"lambda_capacity_edge_{u}_{v}"] = 0.0

                    # Add flow variables f_{s}_{t}_{u}_{v} for all edges
                    for u in range(self.num_nodes):
                        for v in range(self.num_nodes):
                            if u != v:
                                flow_key = f"f_{s}_{t}_{u}_{v}"
                                if flow_key not in enhanced_all_vars:
                                    new_vars[flow_key] = 0.0

                    # Add flow_{s}_{t} variables
                    flow_total_key = f"flow_{s}_{t}"
                    if flow_total_key not in enhanced_all_vars:
                        new_vars[flow_total_key] = 0.0

        # Add lambda_weight_sum if not present
        if "lambda_weight_sum" not in enhanced_all_vars:
            new_vars["lambda_weight_sum"] = 0.0

        # Add missing demand variables to ensure all expected inputs are present
        for s in range(self.num_nodes):
            for t in range(self.num_nodes):
                if s != t:
                    demand_key = f"demand_{s}_{t}"
                    if demand_key not in enhanced_all_vars:
                        new_vars[demand_key] = 0.0

        # Add all new variables at once
        enhanced_all_vars.update(new_vars)

        gradient_start = time.time()
        gradient = self.get_arrow_lagrangian_gradient_wrapper(
            enhanced_all_vars
        )
        gradient_end = time.time()
        if ENABLE_PRINT:
            print(
                f"get_arrow_lagrangian_gradient_wrapper call took {gradient_end - gradient_start:.4f} seconds"
            )

        return {
            "optimal_value": optimal_sol["optimal_total_flow"],
            "flow_values": optimal_sol.get("flow_values", {}),
            "path_flow_values": optimal_sol.get("path_flow_values", {}),
            "gradient": gradient,
            "all_vars": enhanced_all_vars,
        }

    def compute_heuristic_value(self, args_dict):
        """
        Compute the heuristic value for the Arrow problem.
        """
        self.num_compute_heuristic_value_called += 1
        input_dict = args_dict["input_dict"]
        demands = self._extract_demands_from_input_dict(input_dict)
        heuristic_start = time.time()
        heuristic_sol = self.arrow_wrapper(
            demands
        )  # Use wrapper for deterministic behavior
        heuristic_end = time.time()
        # print(f"arrow_wrapper execution took {heuristic_end - heuristic_start:.4f} seconds")
        return {
            "code_path_num": heuristic_sol["code_path_num"],
            "heuristic_value": heuristic_sol["optimal_total_flow"],
            "all_vars": heuristic_sol["all_vars"],
        }

    def compute_lagrangian_gradient(self, args_dict):
        """
        Compute the Lagrangian gradient for the Arrow problem.
        """
        input_dict = args_dict["input_dict"]

        gradient_start = time.time()
        demand_gradient = self.get_arrow_lagrangian_gradient_wrapper(
            input_dict
        )
        gradient_end = time.time()
        if ENABLE_PRINT:
            print(
                f"get_arrow_lagrangian_gradient_wrapper call took {gradient_end - gradient_start:.4f} seconds"
            )

        return demand_gradient

    def compute_lagrangian_value(self, args_dict, give_relaxed_gap=False):
        """
        Compute the Lagrangian value for the Arrow problem.
        """
        input_dict = args_dict["input_dict"]

        lagrangian_start = time.time()
        result = self.get_arrow_lagrangian(input_dict, give_relaxed_gap)
        lagrangian_end = time.time()
        if ENABLE_PRINT:
            print(
                f"get_arrow_lagrangian call took {lagrangian_end - lagrangian_start:.4f} seconds"
            )

        return result

    def compute_relaxed_optimal_value(self, args_dict):
        """
        Compute the relaxed optimal value for the Arrow problem.
        """
        input_dict = args_dict["input_dict"]
        demands = self._extract_demands_from_input_dict(input_dict)

        optimal_start = time.time()
        optimal_sol = self.relaxed_optimal_wrapper(demands)
        optimal_end = time.time()
        if ENABLE_PRINT:
            print(
                f"relaxed_optimal call took {optimal_end - optimal_start:.4f} seconds"
            )

        relaxed_all_vars = optimal_sol["all_vars"]

        return {
            "relaxed_optimal_value": optimal_sol["optimal_total_flow"],
            "relaxed_all_vars": relaxed_all_vars,
        }
    # TODO: needs a more discriptive comment, who sets these thresholds? how do you determine what they should be?
    def get_thresholds(self, relaxed_all_vars):
        """
        Get variable thresholds for the Arrow problem.
        """
        thresholds = {}

        # Demand variables
        for key in relaxed_all_vars:
            if key.startswith("demand_"):
                thresholds[key] = (
                    self.problem_config["min_value"],
                    self.problem_config["max_value"],
                )
            elif key.startswith("weight_"):
                thresholds[key] = (0, 1)  # Weight variables are between 0 and 1
            elif key.startswith("flow_"):
                thresholds[key] = (
                    self.problem_config["min_value"],
                    self.problem_config["max_value"],
                )
            elif key.startswith("f_"):
                thresholds[key] = (
                    0,
                    self.problem_config["max_value"],
                )  # Edge flow variables
            elif key.startswith("lambda_"):
                thresholds[key] = (0, LAMBDA_MAX_VALUE)  # Lagrange multipliers
            elif key.startswith("path_flow_"):
                thresholds[key] = (
                    0,
                    self.problem_config["max_value"],
                )  # Path flow variables

        return thresholds

    def get_decision_to_input_map(self, all_vars):
        """
        Create a mapping of decision variables to their corresponding input variables.
        """
        decision_to_input_map = {}

        # Map flow variables to demand variables
        for var_name in all_vars.keys():
            if var_name.startswith("flow_"):
                # For flow_s_t, the corresponding input is demand_s_t
                parts = var_name.split("_")
                s, t = parts[1], parts[2]
                input_var = f"demand_{s}_{t}"
                decision_to_input_map[var_name] = input_var
            elif var_name.startswith("f_"):
                # For f_s_t_u_v, the corresponding input is demand_s_t
                parts = var_name.split("_")
                s, t = parts[1], parts[2]
                input_var = f"demand_{s}_{t}"
                decision_to_input_map[var_name] = input_var
            elif var_name.startswith("path_flow_"):
                # For path_flow_s_t_..., the corresponding input is demand_s_t
                parts = var_name.split("_")
                s, t = parts[2], parts[3]  # path_flow_s_t_...
                input_var = f"demand_{s}_{t}"
                decision_to_input_map[var_name] = input_var

        return decision_to_input_map

    def is_input_feasible(self, input_dict):
        """
        Check if the input dictionary represents a feasible problem instance.

        For the Arrow problem, we check:
        1. All demand values are non-negative
        2. All demand values are within reasonable bounds
        3. At least one demand exists
        4. All required keys are present
        """
        # Check if there are any demands
        demands = {}
        for key, value in input_dict.items():
            if key.startswith("demand_"):
                parts = key.split("_")
                if len(parts) == 3:  # demand_s_t format
                    try:
                        s, t = int(parts[1]), int(parts[2])
                        demands[(s, t)] = int(value)
                    except ValueError:
                        return False  # Invalid demand key format

        # Check if we have any demands
        if not demands:
            return False

        # Check if all demand values are reasonable
        max_reasonable_demand = self.problem_config["max_value"]
        for (s, t), value in demands.items():
            if value < 0 or value > max_reasonable_demand:
                return False

            # Check if source and destination are different
            if s == t:
                return False

        # Check if we have valid tickets to work with
        if not self.all_tickets:
            return False

        # Check if the topology has enough nodes for the demands
        if demands:
            max_node = max(max(s, t) for (s, t) in demands)
            # Get number of nodes from the first ticket
            if self.all_tickets:
                first_ticket_conv = ticket_to_demand_pinning_format(self.all_tickets[0])
                num_nodes = first_ticket_conv["num_nodes"]
                if max_node >= num_nodes:
                    return False

        return True

    def get_common_header(self, args_dict):
        """
        Generate the common header for Klee programs.
        This provides the basic structure and includes for C programs.
        """
        num_nodes = self.num_nodes
        num_tickets = self.problem_config["num_tickets"]
        num_seeds = len(self.seeds)

        program = f"""
        #include <stdio.h>
        #include <stdlib.h>
        #include <string.h>
        #include <klee/klee.h>

        #define MAX_NODES {num_nodes}
        #define MAX_EDGES (MAX_NODES * (MAX_NODES - 1))
        #define INF 1000000
        #define MAX_TICKETS {num_tickets}
        #define MAX_SEEDS {num_seeds}
        #define WAVELENGTH_CAPACITY 100
        """

        program += """
        typedef struct {
            int from;
            int to;
            int capacity;
        } Edge;

        typedef struct {
            int id;
        } Node;

        typedef struct {
            Node nodes[MAX_NODES];
            Edge edges[MAX_EDGES];
            int node_count;
            int edge_count;
        } Topology;

        typedef struct {
            int from;
            int to;
            int demand;
        } Demand;

        typedef struct {
            char ticket_id[50];
            Edge allocations[MAX_EDGES];
            int allocation_count;
        } LotteryTicket;

        void add_node(Topology* topology, int id) {
            topology->nodes[topology->node_count].id = id;
            topology->node_count++;
        }

        void add_edge(Topology* topology, int from, int to, int capacity) {
            topology->edges[topology->edge_count].from = from;
            topology->edges[topology->edge_count].to = to;
            topology->edges[topology->edge_count].capacity = capacity;
            topology->edge_count++;
        }

        void clear_topology(Topology* topology) {
            topology->node_count = 0;
            topology->edge_count = 0;
        }

        // Simple TE optimization solver (simplified version of optimal_TE)
        int solve_te_optimization(Topology* topology, Demand* demands, int demand_count) {
            int total_flow = 0;
            
            // Simple greedy approach: for each demand, find the best path and allocate flow
            for (int i = 0; i < demand_count; i++) {
                int max_flow_for_demand = 0;
                
                // Find all paths from demands[i].from to demands[i].to
                // For simplicity, we'll use a greedy approach: find the path with highest bottleneck
                
                // Simple path finding: direct edge or 2-hop path
                int direct_capacity = 0;
                int two_hop_capacity = INF;
                
                // Check direct edge
                for (int j = 0; j < topology->edge_count; j++) {
                    if (topology->edges[j].from == demands[i].from && 
                        topology->edges[j].to == demands[i].to) {
                        direct_capacity = topology->edges[j].capacity;
                        break;
                    }
                }
                
                // Check 2-hop path (simplified)
                if (direct_capacity == 0) {
                    int min_capacity = INF;
                    for (int j = 0; j < topology->edge_count; j++) {
                        if (topology->edges[j].from == demands[i].from) {
                            for (int k = 0; k < topology->edge_count; k++) {
                                if (topology->edges[k].from == topology->edges[j].to && 
                                    topology->edges[k].to == demands[i].to) {
                                    int bottleneck = (topology->edges[j].capacity < topology->edges[k].capacity) ? 
                                                   topology->edges[j].capacity : topology->edges[k].capacity;
                                    if (bottleneck < min_capacity) {
                                        min_capacity = bottleneck;
                                    }
                                }
                            }
                        }
                    }
                    if (min_capacity < INF) {
                        two_hop_capacity = min_capacity;
                    }
                }
                
                // Use the better of direct or 2-hop
                int available_capacity = (direct_capacity > two_hop_capacity) ? direct_capacity : two_hop_capacity;
                if (available_capacity == INF) available_capacity = 0;
                
                // Allocate flow (min of demand and capacity)
                int flow = (demands[i].demand < available_capacity) ? demands[i].demand : available_capacity;
                max_flow_for_demand = flow;
                
                total_flow += max_flow_for_demand;
            }
            
            return total_flow;
        }

        // Generate topology from ticket
        void generate_topology_from_ticket(Topology* topology, LotteryTicket* ticket) {
            clear_topology(topology);
            
            // Add all nodes
            for (int i = 0; i < MAX_NODES; i++) {
                add_node(topology, i);
            }
            
            // Add edges from ticket allocations
            for (int i = 0; i < ticket->allocation_count; i++) {
                Edge* edge = &ticket->allocations[i];
                add_edge(topology, edge->from, edge->to, edge->capacity);
            }
        }
        """

        # Add Arrow-specific functions
        program += """
        // Arrow-specific functions for ticket-based optimization
        
        // Function to find the best ticket from a set of selected tickets
        int find_best_ticket_flow(LotteryTicket* selected_tickets[], int num_tickets, Demand* demands, int demand_count) {
            int best_flow = 0;
            
            for (int t = 0; t < num_tickets; t++) {
                Topology temp_topology;
                generate_topology_from_ticket(&temp_topology, selected_tickets[t]);
                int current_flow = solve_te_optimization(&temp_topology, demands, demand_count);
                
                if (current_flow > best_flow) {
                    best_flow = current_flow;
                }
            }
            
            return best_flow;
        }
        
        // Function to process multiple seeds and find the overall best result
        int process_arrow_seeds(LotteryTicket* all_tickets, int num_tickets, Demand* demands, int demand_count, int* selected_indices, int num_seeds) {
            int best_total_flow = 0;
            
            for (int seed_idx = 0; seed_idx < num_seeds; seed_idx++) {
                int ticket_idx = selected_indices[seed_idx];
                if (ticket_idx >= num_tickets) continue;
                
                Topology topology;
                generate_topology_from_ticket(&topology, &all_tickets[ticket_idx]);
                int total_flow = solve_te_optimization(&topology, demands, demand_count);
                
                if (total_flow > best_total_flow) {
                    best_total_flow = total_flow;
                }
            }
            
            return best_total_flow;
        }
        """

        return program

    def get_arrow_program(self, demand_count_dict):
        """
        Generate the Arrow heuristic program logic.
        This follows the same pattern as PoP but for ticket-based selection.
        """
        program = f"""
        // Process Arrow heuristic with pre-selected tickets for each seed
        int best_total_flow = 0;
        
        // For each seed, evaluate the pre-selected tickets
        for (int seed_idx = 0; seed_idx < MAX_SEEDS; seed_idx++) {{
            // Evaluate each ticket for this seed
            for (int ticket_idx = 0; ticket_idx < MAX_TICKETS; ticket_idx++) {{
                Topology topology;
                generate_topology_from_ticket(&topology, &selected_tickets_per_seed[seed_idx][ticket_idx]);
                int total_flow = solve_te_optimization(&topology, demands, {len(demand_count_dict)});
                
                if (total_flow > best_total_flow) {{
                    best_total_flow = total_flow;
                }}
            }}
        }}
        
        return 0;
        }}
        """

        return program

    def generate_heuristic_program(
        self,
        program_type,
        list_of_input_paths_to_exclude=[],
        num_klee_inputs=None,
        path_to_assigned_fixed_points=None,
    ):
        """
        Generate a Klee program for the Arrow problem.

        Args:
            program_type: Type of program to generate
            list_of_input_paths_to_exclude: List of input paths to exclude
            num_klee_inputs: Number of Klee inputs to use
            path_to_assigned_fixed_points: Path to fixed points file
        """
        # Get basic problem information
        num_nodes = self.num_nodes  # Use the actual number of nodes from the problem
        max_flow = self.problem_config["max_value"]

        # Determine which demands to make symbolic
        file_fixed_points = None
        if path_to_assigned_fixed_points:
            with open(path_to_assigned_fixed_points, "r") as f:
                file_fixed_points = json.load(f)
            selected_klee_inputs = [
                name
                for name in self.all_klee_var_names
                if name not in file_fixed_points
            ]
        else:
            if num_klee_inputs is not None:
                num_klee_inputs = min(num_klee_inputs, len(self.all_klee_var_names))
                selected_klee_inputs = random.sample(
                    self.all_klee_var_names, num_klee_inputs
                )
            else:
                selected_klee_inputs = self.all_klee_var_names

        fixed_points = {}

        # Generate the program
        program = self.get_common_header({"num_nodes": num_nodes})
        program += """
        int main() {
        """

        # Add demand variables
        # It's all one path, so it doesn't really matter what we do for the rest of the code, the value matters
        # I want to go through all the self.selected_tickets_per_scenario_per_seed and see what is the minimum, mean, and max value of the demand that is possible through all the tickets
        # import pdb; pdb.set_trace()
        # self.node_to_index
        # self.all_edges
        # self.fiber_edge_constraints
        inverse_node_to_index = {v: k for k, v in self.node_to_index.items()}
        def get_demand_path_capacity_for_ticket(ticket, from_, to_):
            for edge, fiber_allocations in ticket.allocations:
                from_key = inverse_node_to_index[from_]
                to_key = inverse_node_to_index[to_]
                if edge.source == from_key and edge.target == to_key:
                    return edge.capacity
            return max_flow

        def get_edge_capacity_data_for_demand(from_, to_):
            demand_dict = []
            for scenario_str, selected_tickets_per_seed in self.arrow_selected_tickets_per_scenario_per_seed.items():
                for seed, selected_tickets in selected_tickets_per_seed.items():
                    for ticket in selected_tickets:
                        capacity = get_demand_path_capacity_for_ticket(ticket, from_, to_)
                        if capacity is not None:
                            demand_dict.append(capacity)
            return demand_dict

        demand_dict = {}
        for from_ in range(num_nodes):
            for to_ in range(num_nodes):
                if from_ != to_:
                    demand_dict[(from_, to_)] = get_edge_capacity_data_for_demand(from_, to_)

        demand_count = 0
        for from_ in range(num_nodes):
            for to_ in range(num_nodes):
                if from_ != to_:
                    demand_key = f"demand_{from_}_{to_}"
                    try:
                        max_demand = max(demand_dict[(from_, to_)])
                    except:
                        max_demand = max_flow
                    if demand_key in selected_klee_inputs:
                        # value = random.randint(min(max_demand, 100), max_demand)
                        value = max_demand
                        program += f"""
            int {demand_key};
            klee_make_symbolic(&{demand_key}, sizeof({demand_key}), "{demand_key}");
            klee_assume({demand_key} >= {value} && {demand_key} <= {max_flow});
                        """
                    else:
                        if (
                            file_fixed_points is not None
                            and demand_key in file_fixed_points
                        ):
                            value = file_fixed_points[demand_key]
                        else:
                            value = random.randint(min(max_demand, 100), max_demand)
                        fixed_points[demand_key] = value
                        program += f"""
            int {demand_key} = {value};
                        """
                    demand_count += 1

        # Create demand array
        program += f"""
            Demand demands[{demand_count}];
        """

        # Fill demand array
        demand_idx = 0
        for from_ in range(num_nodes):
            for to_ in range(num_nodes):
                if from_ != to_:
                    demand_key = f"demand_{from_}_{to_}"
                    program += f"""
            demands[{demand_idx}] = (Demand){{.from = {from_}, .to = {to_}, .demand = {demand_key}}};
                    """
                    demand_idx += 1

        # Pre-select tickets for each seed (matching Python logic)
        import numpy as np

        selected_tickets_per_seed = []

        # Check if we have any tickets available
        if not self.all_tickets:
            raise ValueError(
                "No tickets available — cannot generate heuristic program. Please check the topology configuration and ensure feasible solutions exist."
            )

        for seed in self.seeds:
            np.random.seed(seed)
            selected_indices = np.random.choice(
                range(len(self.all_tickets)),
                min(self.num_tickets, len(self.all_tickets)),
                replace=False,
            )
            selected_tickets = [self.all_tickets[i] for i in selected_indices]
            selected_tickets_per_seed.append(selected_tickets)

        # Define selected tickets for each seed
        program += f"""
            // Define selected tickets for each seed (pre-selected in Python)
            LotteryTicket selected_tickets_per_seed[MAX_SEEDS][MAX_TICKETS] = {{
        """

        for seed_idx, selected_tickets in enumerate(selected_tickets_per_seed):
            program += f"""
                {{ // Seed {seed_idx}
        """
            for ticket_idx, ticket in enumerate(selected_tickets):
                program += f"""
                    {{ // Ticket {ticket_idx}: {ticket.ticket_id}
                        .ticket_id = "{ticket.ticket_id}",
                        .allocation_count = {len(ticket.allocations)},
                        .allocations = {{
        """
                for j, (edge, fiber_allocations) in enumerate(ticket.allocations):
                    # Convert edge to from/to format
                    edge_str = str(edge)
                    if edge_str.startswith("edge_"):
                        parts = edge_str.split("_")
                        from_node = parts[1]
                        to_node = parts[2]
                    else:
                        from_node = "0"
                        to_node = "1"

                    # Calculate capacity from fiber allocations using proper multi-hop path analysis
                    capacity = get_capacity_from_fiber_allocations(
                        fiber_allocations, edge.source, edge.target
                    )

                    program += f"""
                            {{.from = {from_node}, .to = {to_node}, .capacity = {capacity}}},
                    """

                program += """
                        }
                    },
        """
            program += """
                },
        """

        program += """
            };
        """

        # Create demand count dictionary for the Arrow program
        demand_count_dict = {}
        demand_idx = 0
        for from_ in range(num_nodes):
            for to_ in range(num_nodes):
                if from_ != to_:
                    demand_count_dict[(from_, to_)] = demand_idx
                    demand_idx += 1

        # Generate Arrow heuristic program logic
        program += self.get_arrow_program(demand_count_dict)
        return {"program": program, "fixed_points": fixed_points}