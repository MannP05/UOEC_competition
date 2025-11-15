import json
import math
import time
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from copy import deepcopy
import heapq

@dataclass
class DeliveryPoint:
    """Represents a delivery location with time window constraints"""
    id: int
    x: float
    y: float
    earliest: float  # Earliest arrival time
    latest: float    # Latest arrival time
    service_time: float = 1  # Time spent at location (minutes)

@dataclass
class TrafficCondition:
    """Represents traffic conditions between two points"""
    from_id: int
    to_id: int
    speed_factor: float  # 1.0 = normal, 0.5 = slow (congestion), 2.0 = fast
    delay: float = 0.0   # Additional delay in minutes

class RouteOptimizer:
    """
    Main route optimization engine using:
    - Nearest Neighbor construction heuristic with ALL nodes guarantee
    - 2-opt local search improvement
    - Time window feasibility checking
    - Dynamic traffic adaptation
    """
    
    def __init__(self, 
                 delivery_points: List[DeliveryPoint], 
                 traffic_conditions: Dict[Tuple[int, int], TrafficCondition],
                 depot_id: int = 0,
                 base_speed: float = 40.0):  # km/h
        
        self.points = {p.id: p for p in delivery_points}
        self.traffic = traffic_conditions
        self.depot_id = depot_id
        self.base_speed = base_speed
        
        # Caching for performance
        self.distance_cache = {}
        self.time_cache = {}
        
        # Pre-calculate all distances
        self._precompute_distances()
    
    def _precompute_distances(self):
        """Pre-compute all pairwise distances for faster lookup"""
        point_list = list(self.points.values())
        for i, p1 in enumerate(point_list):
            for p2 in point_list[i:]:
                dist = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                self.distance_cache[(p1.id, p2.id)] = dist
                self.distance_cache[(p2.id, p1.id)] = dist
    
    def get_distance(self, id1: int, id2: int) -> float:
        """Get Euclidean distance between two points"""
        return self.distance_cache.get((id1, id2), 0.0)
    
    def calculate_travel_time(self, from_id: int, to_id: int, current_time: float) -> float:
        """Calculate travel time considering traffic conditions"""
        distance = self.get_distance(from_id, to_id)
        
        # Check for specific traffic condition
        traffic_key = (from_id, to_id)
        if traffic_key in self.traffic:
            traffic = self.traffic[traffic_key]
            speed = self.base_speed * traffic.speed_factor
            travel_time = (distance / speed) * 60 + traffic.delay  # Convert to minutes
        else:
            travel_time = (distance / self.base_speed) * 60
        
        return travel_time
    
    def evaluate_route(self, route: List[int], start_time: float = 0.0, 
                      penalize_violations: bool = False) -> Tuple[List[float], float, bool]:
        """
        Evaluate a route and return arrival times, total time, and feasibility
        Returns: (arrival_times, total_completion_time, is_feasible)
        
        If penalize_violations=True, allows time window violations but adds penalty
        """
        if not route:
            return [], 0.0, True
        
        arrival_times = []
        current_time = start_time
        current_id = self.depot_id
        is_feasible = True
        penalty = 0.0
        
        for node_id in route:
            # Calculate travel time from current position
            travel_time = self.calculate_travel_time(current_id, node_id, current_time)
            current_time += travel_time
            
            point = self.points[node_id]
            
            # Wait if we arrive too early
            if current_time < point.earliest:
                current_time = point.earliest
            
            # Check if we're too late
            if current_time > point.latest:
                if penalize_violations:
                    penalty += (current_time - point.latest) * 100  # Heavy penalty
                    is_feasible = False
                else:
                    return arrival_times, float('inf'), False
            
            arrival_times.append(current_time)
            
            # Add service time
            current_time += point.service_time
            current_id = node_id
        
        total_time = current_time + penalty
        return arrival_times, total_time, is_feasible
    
    def nearest_neighbor_construction(self, start_time: float = 0.0, 
                                     must_visit_all: bool = True) -> List[int]:
        """
        Build initial route using nearest neighbor - GUARANTEES ALL NODES VISITED
        """
        unvisited = set(self.points.keys()) - {self.depot_id}
        route = []
        current_id = self.depot_id
        current_time = start_time
        
        while unvisited:
            best_next = None
            best_score = float('inf')
            best_arrival = None
            best_is_feasible = False
            
            for next_id in unvisited:
                next_point = self.points[next_id]
                travel_time = self.calculate_travel_time(current_id, next_id, current_time)
                arrival_time = current_time + travel_time
                
                # Adjust for time window
                actual_arrival = max(arrival_time, next_point.earliest)
                
                # Check feasibility
                is_feasible = actual_arrival <= next_point.latest
                
                # Calculate score
                distance_score = self.get_distance(current_id, next_id)
                
                if is_feasible:
                    # Prefer feasible nodes with tight time windows
                    time_urgency = next_point.latest - actual_arrival
                    slack_penalty = 1.0 / (time_urgency + 1)
                    score = distance_score * (1 + slack_penalty)
                else:
                    # For infeasible nodes, use distance + lateness penalty
                    lateness = actual_arrival - next_point.latest
                    score = distance_score + lateness * 10
                
                # Prefer feasible over infeasible
                if best_next is None:
                    best_next = next_id
                    best_score = score
                    best_arrival = actual_arrival
                    best_is_feasible = is_feasible
                elif is_feasible and not best_is_feasible:
                    # Always prefer feasible over infeasible
                    best_next = next_id
                    best_score = score
                    best_arrival = actual_arrival
                    best_is_feasible = is_feasible
                elif is_feasible == best_is_feasible and score < best_score:
                    # Among same feasibility, prefer better score
                    best_next = next_id
                    best_score = score
                    best_arrival = actual_arrival
                    best_is_feasible = is_feasible
            
            # ALWAYS add a node (guarantee all nodes visited)
            if best_next is not None:
                route.append(best_next)
                unvisited.remove(best_next)
                current_id = best_next
                current_time = best_arrival + self.points[best_next].service_time
        
        return route
    
    def cheapest_insertion(self, start_time: float = 0.0) -> List[int]:
        """
        Alternative construction: cheapest insertion heuristic
        Guarantees all nodes are visited
        """
        unvisited = set(self.points.keys()) - {self.depot_id}
        
        if not unvisited:
            return []
        
        # Start with nearest node to depot
        first_node = min(unvisited, key=lambda x: self.get_distance(self.depot_id, x))
        route = [first_node]
        unvisited.remove(first_node)
        
        # Insert remaining nodes
        while unvisited:
            best_node = None
            best_position = 0
            best_cost = float('inf')
            
            for node in unvisited:
                # Try inserting at each position
                for pos in range(len(route) + 1):
                    test_route = route[:pos] + [node] + route[pos:]
                    _, cost, _ = self.evaluate_route(test_route, start_time, penalize_violations=True)
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_node = node
                        best_position = pos
            
            # Insert the best node
            if best_node is not None:
                route.insert(best_position, best_node)
                unvisited.remove(best_node)
        
        return route
    
    def two_opt_improvement(self, route: List[int], start_time: float = 0.0, 
                           max_iterations: int = 500) -> List[int]:
        """
        Improve route using 2-opt local search
        """
        if len(route) <= 3:
            return route
        
        best_route = route[:]
        _, best_time, _ = self.evaluate_route(best_route, start_time, penalize_violations=True)
        
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for i in range(len(best_route) - 1):
                for j in range(i + 2, min(i + 20, len(best_route))):  # Limited neighborhood
                    # Reverse segment between i and j
                    new_route = best_route[:i+1] + best_route[i+1:j+1][::-1] + best_route[j+1:]
                    
                    _, new_time, _ = self.evaluate_route(new_route, start_time, penalize_violations=True)
                    
                    if new_time < best_time:
                        best_route = new_route
                        best_time = new_time
                        improved = True
                        break
                
                if improved:
                    break
        
        return best_route
    
    def or_opt_improvement(self, route: List[int], start_time: float = 0.0) -> List[int]:
        """
        Or-opt improvement: relocate sequences of 1, 2, or 3 consecutive nodes
        """
        if len(route) <= 3:
            return route
        
        best_route = route[:]
        _, best_time, _ = self.evaluate_route(best_route, start_time, penalize_violations=True)
        
        improved = True
        while improved:
            improved = False
            
            # Try moving sequences of length 1, 2, 3
            for seq_len in [1, 2, 3]:
                if len(best_route) < seq_len + 1:
                    continue
                
                for i in range(len(best_route) - seq_len + 1):
                    # Extract sequence
                    sequence = best_route[i:i+seq_len]
                    remaining = best_route[:i] + best_route[i+seq_len:]
                    
                    # Try inserting at each position
                    for j in range(len(remaining) + 1):
                        new_route = remaining[:j] + sequence + remaining[j:]
                        _, new_time, _ = self.evaluate_route(new_route, start_time, penalize_violations=True)
                        
                        if new_time < best_time:
                            best_route = new_route
                            best_time = new_time
                            improved = True
                            break
                    
                    if improved:
                        break
                
                if improved:
                    break
        
        return best_route
    
    def optimize_route(self, start_time: float = 0.0, 
                      time_limit: float = 4.5) -> Dict:
        """
        Main optimization function with multiple strategies
        GUARANTEES ALL NODES ARE VISITED
        """
        optimization_start = time.time()
        
        # Try both construction methods
        route1 = self.nearest_neighbor_construction(start_time)
        _, time1, _ = self.evaluate_route(route1, start_time, penalize_violations=True)
        
        # For smaller instances, try cheapest insertion too
        if len(self.points) <= 50 and (time.time() - optimization_start) < time_limit * 0.3:
            route2 = self.cheapest_insertion(start_time)
            _, time2, _ = self.evaluate_route(route2, start_time, penalize_violations=True)
            
            if time2 < time1:
                route = route2
            else:
                route = route1
        else:
            route = route1
        
        # Verify all nodes are in route
        expected_nodes = set(self.points.keys()) - {self.depot_id}
        actual_nodes = set(route)
        
        if expected_nodes != actual_nodes:
            print(f"⚠️  WARNING: Missing nodes: {expected_nodes - actual_nodes}")
            # Add missing nodes at the end
            missing = list(expected_nodes - actual_nodes)
            route.extend(missing)
        
        arrival_times, total_time, feasible = self.evaluate_route(route, start_time, penalize_violations=False)
        
        # Phase 2: 2-opt Improvement
        elapsed = time.time() - optimization_start
        if elapsed < time_limit * 0.6:
            route = self.two_opt_improvement(route, start_time)
            arrival_times, total_time, feasible = self.evaluate_route(route, start_time, penalize_violations=False)
        
        # Phase 3: Or-opt improvement for smaller instances
        elapsed = time.time() - optimization_start
        if elapsed < time_limit * 0.8 and len(route) < 50:
            route = self.or_opt_improvement(route, start_time)
            arrival_times, total_time, feasible = self.evaluate_route(route, start_time, penalize_violations=False)
        
        computation_time = time.time() - optimization_start
        
        # Final verification
        visited_count = len(route)
        expected_count = len(self.points) - 1  # Excluding depot
        
        # Prepare detailed output
        result = {
            "route": [self.depot_id] + route,
            "delivery_sequence": route,
            "delivery_times": {},
            "time_windows": {},
            "total_time_minutes": round(total_time, 2),
            "computation_time_seconds": round(computation_time, 4),
            "num_deliveries": len(route),
            "expected_deliveries": expected_count,
            "all_nodes_visited": visited_count == expected_count,
            "feasible": feasible,
            "depot_id": self.depot_id
        }
        
        for i, node_id in enumerate(route):
            point = self.points[node_id]
            if i < len(arrival_times):
                result["delivery_times"][f"node_{node_id}"] = round(arrival_times[i], 2)
                result["time_windows"][f"node_{node_id}"] = {
                    "earliest": point.earliest,
                    "latest": point.latest,
                    "arrival": round(arrival_times[i], 2),
                    "slack": round(point.latest - arrival_times[i], 2),
                    "violated": arrival_times[i] > point.latest
                }
        
        return result
    
    def update_traffic(self, new_traffic: Dict[Tuple[int, int], TrafficCondition]):
        """Update traffic conditions for real-time adaptation"""
        self.traffic = new_traffic
        self.time_cache.clear()
        print(f"Traffic updated: {len(new_traffic)} conditions loaded")
    
    def reoptimize_from_position(self, current_node: int, 
                                 completed: Set[int], 
                                 current_time: float) -> Dict:
        """
        Re-optimize remaining route from current position
        Used for dynamic re-routing during delivery
        """
        # Create temporary instance with remaining points
        remaining_points = [p for p in self.points.values() 
                          if p.id not in completed and p.id != self.depot_id]
        
        if not remaining_points:
            return {"route": [current_node], "delivery_times": {}, 
                   "total_time_minutes": 0, "num_deliveries": 0}
        
        # Temporarily adjust depot
        temp_optimizer = RouteOptimizer(
            delivery_points=remaining_points,
            traffic_conditions=self.traffic,
            depot_id=current_node,
            base_speed=self.base_speed
        )
        
        result = temp_optimizer.optimize_route(current_time)
        result["reoptimized"] = True
        result["reoptimization_point"] = current_node
        
        return result


def load_from_json(json_file: str) -> Tuple[List[DeliveryPoint], Dict, Dict]:
    """Load delivery data from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Parse delivery points
    delivery_points = []
    for point in data.get('delivery_points', []):
        dp = DeliveryPoint(
            id=point['id'],
            x=point['x'],
            y=point['y'],
            earliest=point.get('earliest', 0),
            latest=point.get('latest', 1440),
            service_time=point.get('service_time', 5)
        )
        delivery_points.append(dp)
    
    # Parse traffic conditions
    traffic_conditions = {}
    for traffic in data.get('traffic_conditions', []):
        key = (traffic['from_id'], traffic['to_id'])
        tc = TrafficCondition(
            from_id=traffic['from_id'],
            to_id=traffic['to_id'],
            speed_factor=traffic.get('speed_factor', 1.0),
            delay=traffic.get('delay', 0.0)
        )
        traffic_conditions[key] = tc
    
    config = data.get('config', {
        'depot_id': 0,
        'base_speed': 40.0,
        'start_time': 0
    })
    
    return delivery_points, traffic_conditions, config


def save_result_to_json(result: Dict, output_file: str):
    """Save optimization result to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {output_file}")


def create_sample_input(filename: str, num_points: int = 20):
    """Create a sample input JSON file for testing - WITH FEASIBLE TIME WINDOWS"""
    import random
    random.seed(42)
    
    city_size = 50  # km x km city
    
    delivery_points = []
    
    # Depot at center
    delivery_points.append({
        "id": 0,
        "x": 0.0,
        "y": 0.0,
        "earliest": 0,
        "latest": 2000,  # Extended for feasibility
        "service_time": 0
    })
    
    # Generate random delivery points with RELAXED time windows
    for i in range(1, num_points + 1):
        x = round(random.uniform(-city_size/2, city_size/2), 2)
        y = round(random.uniform(-city_size/2, city_size/2), 2)
        
        # More relaxed time windows to ensure feasibility
        earliest = random.uniform(0, 200)
        time_window_size = random.uniform(300, 800)  # Larger windows
        
        delivery_points.append({
            "id": i,
            "x": x,
            "y": y,
            "earliest": round(earliest, 2),
            "latest": round(earliest + time_window_size, 2),
            "service_time": round(random.uniform(5, 15), 2)
        })
    
    # Generate random traffic conditions
    traffic_conditions = []
    num_traffic = min(50, num_points * 2)
    
    for _ in range(num_traffic):
        from_id = random.randint(0, num_points)
        to_id = random.randint(0, num_points)
        
        if from_id != to_id:
            traffic_conditions.append({
                "from_id": from_id,
                "to_id": to_id,
                "speed_factor": round(random.uniform(0.7, 1.3), 2),
                "delay": round(random.uniform(0, 5), 2)
            })
    
    data = {
        "delivery_points": delivery_points,
        "traffic_conditions": traffic_conditions,
        "config": {
            "depot_id": 0,
            "base_speed": 40.0,
            "start_time": 0
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Sample input file created: {filename}")
    print(f"  - {len(delivery_points)} delivery points (including depot)")
    print(f"  - {num_points} deliveries required")
    print(f"  - {len(traffic_conditions)} traffic conditions")


def demonstrate_dynamic_reoptimization(optimizer: RouteOptimizer, initial_result: Dict):
    """
    Demonstrate dynamic re-optimization capability
    Simulates traffic change mid-route
    """
    print("\n" + "="*60)
    print("DYNAMIC RE-OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    route = initial_result['delivery_sequence']
    
    if len(route) < 5:
        print("Route too short for demonstration")
        return
    
    # Simulate: completed 30% of deliveries
    completed_count = max(1, len(route) // 3)
    completed = set(route[:completed_count])
    current_node = route[completed_count - 1]
    current_time = initial_result['delivery_times'][f'node_{current_node}']
    
    print(f"\nSimulating delivery in progress...")
    print(f"  Completed deliveries: {completed_count}/{len(route)}")
    print(f"  Current position: Node {current_node}")
    print(f"  Current time: {current_time:.2f} minutes")
    
    # Simulate traffic change
    print(f"\n⚠️  Traffic conditions changed!")
    import random
    new_traffic = {}
    for (from_id, to_id), tc in optimizer.traffic.items():
        # Randomly worsen or improve traffic
        new_factor = tc.speed_factor * random.uniform(0.7, 1.3)
        new_traffic[(from_id, to_id)] = TrafficCondition(
            from_id=from_id,
            to_id=to_id,
            speed_factor=new_factor,
            delay=tc.delay * random.uniform(0.8, 1.5)
        )
    
    optimizer.update_traffic(new_traffic)
    
    # Re-optimize
    print(f"\nRe-optimizing remaining route...")
    new_result = optimizer.reoptimize_from_position(current_node, completed, current_time)
    
    print(f"\n✓ Re-optimization complete!")
    print(f"  New route: {new_result['route'][:10]}..." if len(new_result['route']) > 10 else f"  New route: {new_result['route']}")
    print(f"  Remaining deliveries: {new_result['num_deliveries']}")
    print(f"  Estimated completion: {new_result['total_time_minutes']:.2f} minutes")


def main():
    """Main execution function"""
    print("="*60)
    print("REAL-TIME ROUTE OPTIMIZATION SYSTEM")
    print("="*60)
    
    input_file = "delivery_w0nodes.json"
    output_file = "optimized_route.json"
    
    # Create sample input if file doesn't exist
    try:
        with open(input_file, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"\nInput file not found. Creating sample data...")
        num_points = int(input("Enter number of delivery points (default 20, max 100): ") or "20")
        num_points = min(max(num_points, 5), 100)
        create_sample_input(input_file, num_points)
        print()
    
    # Load data
    print(f"\nLoading data from {input_file}...")
    delivery_points, traffic_conditions, config = load_from_json(input_file)
    
    print(f"✓ Loaded {len(delivery_points)} delivery points (including depot)")
    print(f"✓ Deliveries required: {len(delivery_points) - 1}")
    print(f"✓ Loaded {len(traffic_conditions)} traffic conditions")
    
    # Create optimizer
    print(f"\nInitializing optimizer...")
    optimizer = RouteOptimizer(
        delivery_points=delivery_points,
        traffic_conditions=traffic_conditions,
        depot_id=config.get('depot_id', 0),
        base_speed=config.get('base_speed', 40.0)
    )
    
    # Run optimization
    print(f"\nOptimizing route...")
    start_time = config.get('start_time', 0)
    result = optimizer.optimize_route(start_time)
    
    # Display results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    print(f"\n📊 Statistics:")
    print(f"   Expected deliveries: {result['expected_deliveries']}")
    print(f"   Actual deliveries: {result['num_deliveries']}")
    print(f"   All nodes visited: {'✓ YES' if result['all_nodes_visited'] else '✗ NO'}")
    print(f"   Total time: {result['total_time_minutes']:.2f} minutes")
    print(f"   Time windows feasible: {'✓ Yes' if result['feasible'] else '✗ No (some violations)'}")
    print(f"   Computation time: {result['computation_time_seconds']:.4f} seconds")
    
    print(f"\n📍 Route Sequence (first 20 nodes):")
    route_display = result['route'][:21]
    print(f"   {' → '.join(map(str, route_display))}" + 
          (f" → ... ({len(result['route']) - 21} more)" if len(result['route']) > 21 else ""))
    
    print(f"\n📅 Delivery Schedule (first 10):")
    violations = 0
    for i, node_id in enumerate(result['delivery_sequence'][:10]):
        info = result['time_windows'][f'node_{node_id}']
        violation_mark = "⚠️" if info.get('violated', False) else "✓"
        print(f"   {violation_mark} Node {node_id}: "
              f"Arrive {info['arrival']:.1f} min "
              f"(Window: {info['earliest']:.0f}-{info['latest']:.0f}, "
              f"Slack: {info['slack']:.1f} min)")
        if info.get('violated', False):
            violations += 1
    
    if len(result['delivery_sequence']) > 10:
        print(f"   ... and {len(result['delivery_sequence']) - 10} more deliveries")
    
    # Count total violations
    total_violations = sum(1 for info in result['time_windows'].values() if info.get('violated', False))
    if total_violations > 0:
        print(f"\n⚠️  Total time window violations: {total_violations}/{result['num_deliveries']}")
    
    # Save results
    save_result_to_json(result, output_file)
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()