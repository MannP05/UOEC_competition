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
    service_time: float = 5.0  # Time spent at location (minutes)

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
    - Nearest Neighbor construction heuristic
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
    
    def evaluate_route(self, route: List[int], start_time: float = 0.0) -> Tuple[List[float], float, bool]:
        """
        Evaluate a route and return arrival times, total time, and feasibility
        Returns: (arrival_times, total_completion_time, is_feasible)
        """
        if not route:
            return [], 0.0, True
        
        arrival_times = []
        current_time = start_time
        current_id = self.depot_id
        
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
                return arrival_times, float('inf'), False
            
            arrival_times.append(current_time)
            
            # Add service time
            current_time += point.service_time
            current_id = node_id
        
        return arrival_times, current_time, True
    
    def nearest_neighbor_construction(self, start_time: float = 0.0) -> List[int]:
        """
        Build initial route using nearest neighbor with time window awareness
        """
        unvisited = set(self.points.keys()) - {self.depot_id}
        route = []
        current_id = self.depot_id
        current_time = start_time
        
        while unvisited:
            best_next = None
            best_score = float('inf')
            best_arrival = None
            
            for next_id in unvisited:
                next_point = self.points[next_id]
                travel_time = self.calculate_travel_time(current_id, next_id, current_time)
                arrival_time = current_time + travel_time
                
                # Adjust for time window
                actual_arrival = max(arrival_time, next_point.earliest)
                
                # Skip if we'd be late
                if actual_arrival > next_point.latest:
                    continue
                
                # Scoring: prioritize closer points with tighter time windows
                distance_score = self.get_distance(current_id, next_id)
                time_urgency = next_point.latest - actual_arrival
                slack_penalty = 1.0 / (time_urgency + 1)
                
                score = distance_score * (1 + slack_penalty)
                
                if score < best_score:
                    best_score = score
                    best_next = next_id
                    best_arrival = actual_arrival
            
            # If no feasible next point found, try earliest deadline first
            if best_next is None:
                candidates = sorted(unvisited, 
                                  key=lambda x: self.points[x].latest)
                for next_id in candidates:
                    next_point = self.points[next_id]
                    travel_time = self.calculate_travel_time(current_id, next_id, current_time)
                    arrival_time = max(current_time + travel_time, next_point.earliest)
                    
                    if arrival_time <= next_point.latest:
                        best_next = next_id
                        best_arrival = arrival_time
                        break
            
            if best_next is None:
                # Cannot find feasible continuation
                break
            
            route.append(best_next)
            unvisited.remove(best_next)
            current_id = best_next
            current_time = best_arrival + self.points[best_next].service_time
        
        return route
    
    def two_opt_improvement(self, route: List[int], start_time: float = 0.0, 
                           max_iterations: int = 500) -> List[int]:
        """
        Improve route using 2-opt local search
        """
        if len(route) <= 3:
            return route
        
        best_route = route[:]
        _, best_time, feasible = self.evaluate_route(best_route, start_time)
        
        if not feasible:
            return best_route
        
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for i in range(len(best_route) - 1):
                for j in range(i + 2, min(i + 20, len(best_route))):  # Limited neighborhood
                    # Reverse segment between i and j
                    new_route = best_route[:i+1] + best_route[i+1:j+1][::-1] + best_route[j+1:]
                    
                    _, new_time, feasible = self.evaluate_route(new_route, start_time)
                    
                    if feasible and new_time < best_time:
                        best_route = new_route
                        best_time = new_time
                        improved = True
                        break
                
                if improved:
                    break
        
        return best_route
    
    def insertion_improvement(self, route: List[int], start_time: float = 0.0) -> List[int]:
        """
        Try to improve route by removing and reinserting nodes
        """
        if len(route) <= 2:
            return route
        
        best_route = route[:]
        _, best_time, _ = self.evaluate_route(best_route, start_time)
        
        for i in range(len(route)):
            # Remove node i
            node = route[i]
            temp_route = route[:i] + route[i+1:]
            
            # Try inserting it at different positions
            for j in range(len(temp_route) + 1):
                new_route = temp_route[:j] + [node] + temp_route[j:]
                _, new_time, feasible = self.evaluate_route(new_route, start_time)
                
                if feasible and new_time < best_time:
                    best_route = new_route
                    best_time = new_time
        
        return best_route
    
    def optimize_route(self, start_time: float = 0.0, 
                      time_limit: float = 4.5) -> Dict:
        """
        Main optimization function with multiple strategies
        """
        optimization_start = time.time()
        
        # Phase 1: Construction
        route = self.nearest_neighbor_construction(start_time)
        arrival_times, total_time, feasible = self.evaluate_route(route, start_time)
        
        if not feasible:
            print("Warning: Initial route is infeasible!")
        
        # Phase 2: Improvement (if time permits)
        elapsed = time.time() - optimization_start
        if elapsed < time_limit * 0.6:
            route = self.two_opt_improvement(route, start_time)
            arrival_times, total_time, feasible = self.evaluate_route(route, start_time)
        
        # Phase 3: Further refinement
        elapsed = time.time() - optimization_start
        if elapsed < time_limit * 0.8 and len(route) < 50:
            route = self.insertion_improvement(route, start_time)
            arrival_times, total_time, feasible = self.evaluate_route(route, start_time)
        
        computation_time = time.time() - optimization_start
        
        # Prepare detailed output
        result = {
            "route": [self.depot_id] + route,
            "delivery_sequence": route,
            "delivery_times": {},
            "time_windows": {},
            "total_time_minutes": round(total_time, 2),
            "computation_time_seconds": round(computation_time, 4),
            "num_deliveries": len(route),
            "feasible": feasible,
            "depot_id": self.depot_id
        }
        
        for i, node_id in enumerate(route):
            point = self.points[node_id]
            result["delivery_times"][f"node_{node_id}"] = round(arrival_times[i], 2)
            result["time_windows"][f"node_{node_id}"] = {
                "earliest": point.earliest,
                "latest": point.latest,
                "arrival": round(arrival_times[i], 2),
                "slack": round(point.latest - arrival_times[i], 2)
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
    """Create a sample input JSON file for testing"""
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
        "latest": 1440,
        "service_time": 0
    })
    
    # Generate random delivery points
    for i in range(1, num_points):
        earliest = random.uniform(0, 300)
        time_window_size = random.uniform(120, 400)
        
        delivery_points.append({
            "id": i,
            "x": round(random.uniform(-city_size/2, city_size/2), 2),
            "y": round(random.uniform(-city_size/2, city_size/2), 2),
            "earliest": round(earliest, 2),
            "latest": round(earliest + time_window_size, 2),
            "service_time": round(random.uniform(5, 20), 2)
        })
    
    # Generate random traffic conditions
    traffic_conditions = []
    num_traffic = min(50, num_points * 3)
    
    for _ in range(num_traffic):
        from_id = random.randint(0, num_points - 1)
        to_id = random.randint(0, num_points - 1)
        
        if from_id != to_id:
            traffic_conditions.append({
                "from_id": from_id,
                "to_id": to_id,
                "speed_factor": round(random.uniform(0.6, 1.4), 2),
                "delay": round(random.uniform(0, 8), 2)
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
    print(f"  - {len(delivery_points)} delivery points")
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
    print(f"  New route: {new_result['route']}")
    print(f"  Remaining deliveries: {new_result['num_deliveries']}")
    print(f"  Estimated completion: {new_result['total_time_minutes']:.2f} minutes")


def main():
    """Main execution function"""
    print("="*60)
    print("REAL-TIME ROUTE OPTIMIZATION SYSTEM")
    print("="*60)
    
    input_file = "delivery_1000nodes.json"
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
    
    print(f"✓ Loaded {len(delivery_points)} delivery points")
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
    print(f"\n📍 Route Sequence:")
    print(f"   {' → '.join(map(str, result['route']))}")
    print(f"\n📊 Statistics:")
    print(f"   Total deliveries: {result['num_deliveries']}")
    print(f"   Total time: {result['total_time_minutes']:.2f} minutes")
    print(f"   Feasible: {'✓ Yes' if result['feasible'] else '✗ No'}")
    print(f"   Computation time: {result['computation_time_seconds']:.4f} seconds")
    
    print(f"\n📅 Delivery Schedule:")
    for node_id in result['delivery_sequence'][:10]:  # Show first 10
        info = result['time_windows'][f'node_{node_id}']
        print(f"   Node {node_id}: "
              f"Arrive {info['arrival']:.1f} min "
              f"(Window: {info['earliest']:.0f}-{info['latest']:.0f}, "
              f"Slack: {info['slack']:.1f} min)")
    
    if len(result['delivery_sequence']) > 10:
        print(f"   ... and {len(result['delivery_sequence']) - 10} more deliveries")
    
    # Save results
    save_result_to_json(result, output_file)
    
    # Demonstrate dynamic re-optimization
    demonstrate_dynamic_reoptimization(optimizer, result)
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()