import json
import math
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass

# ============================================================================
# DATA Classes 
# ============================================================================

@dataclass
class DeliveryPoint:
    id: int
    x: float
    y: float
    earliest: float
    latest: float
    service_time: float = 5.0

@dataclass
class TrafficCondition:
    from_id: int
    to_id: int
    speed_factor: float
    delay: float = 0.0


# ============================================================================
# ROUTE OPTIMIZATION CLASS 
# ============================================================================
# this class implements the methods such as fast nearest neighbour and clark wrights savings algorithm

class FastRouteOptimizer:
    
    def __init__(self, delivery_points, traffic_conditions, depot_id=0, base_speed=40.0):
        self.points = {p.id: p for p in delivery_points}
        self.traffic = traffic_conditions
        self.depot_id = depot_id
        self.base_speed = base_speed
        self.distance_matrix = {}
        self._precompute_distances()
        self.n = len(delivery_points) - 1
    
    def _precompute_distances(self):
        ids = list(self.points.keys())
        for i in ids:
            p1 = self.points[i]
            for j in ids:
                if i <= j:
                    p2 = self.points[j]
                    dist = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
                    self.distance_matrix[(i, j)] = dist
                    self.distance_matrix[(j, i)] = dist
    
    def get_distance(self, id1, id2):
        return self.distance_matrix.get((id1, id2), 0.0)
    
    def calculate_travel_time(self, from_id, to_id):
        distance = self.get_distance(from_id, to_id)
        
        traffic_key = (from_id, to_id)
        if traffic_key in self.traffic:
            traffic = self.traffic[traffic_key]
            speed = self.base_speed * traffic.speed_factor
            travel_time = (distance / speed) * 60 + traffic.delay
        else:
            travel_time = (distance / self.base_speed) * 60
        
        return travel_time
    
    def evaluate_route_fast(self, route, start_time=0.0):
        if not route:
            return 0.0, True, []
        
        arrival_times = []
        current_time = start_time
        current_id = self.depot_id
        is_feasible = True
        
        for node_id in route:
            travel_time = self.calculate_travel_time(current_id, node_id)
            current_time += travel_time
            
            point = self.points[node_id]
            
            if current_time < point.earliest:
                current_time = point.earliest
            
            if current_time > point.latest:
                is_feasible = False
            
            arrival_times.append(current_time)
            current_time += point.service_time
            current_id = node_id
        
        return current_time, is_feasible, arrival_times
    
    def fast_nearest_neighbor(self, start_time=0.0):
        unvisited = set(self.points.keys()) - {self.depot_id}
        route = []
        current_id = self.depot_id
        current_time = start_time
        
        while unvisited:
            best_next = None
            best_score = float('inf')
            best_arrival = None
            
            for next_id in unvisited:
                point = self.points[next_id]
                travel_time = self.calculate_travel_time(current_id, next_id)
                arrival_time = max(current_time + travel_time, point.earliest)
                
                distance = self.get_distance(current_id, next_id)
                
                if arrival_time <= point.latest:
                    urgency = 1.0 / (point.latest - arrival_time + 1)
                    score = distance * (1 + urgency)
                else:
                    lateness = arrival_time - point.latest
                    score = distance + lateness * 1000
                
                if score < best_score:
                    best_score = score
                    best_next = next_id
                    best_arrival = arrival_time
            
            if best_next is not None:
                route.append(best_next)
                unvisited.remove(best_next)
                current_id = best_next
                current_time = best_arrival + self.points[best_next].service_time
        
        return route
    
    def savings_algorithm(self, start_time=0.0):
        nodes = list(self.points.keys())
        nodes.remove(self.depot_id)
        
        savings = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                ni, nj = nodes[i], nodes[j]
                save = (self.get_distance(self.depot_id, ni) + 
                       self.get_distance(self.depot_id, nj) - 
                       self.get_distance(ni, nj))
                savings.append((save, ni, nj))
        
        savings.sort(reverse=True, key=lambda x: x[0])
        
        routes = {node: [node] for node in nodes}
        route_of = {node: node for node in nodes}
        
        for save, ni, nj in savings:
            ri, rj = route_of[ni], route_of[nj]
            
            if ri != rj:
                route_i = routes[ri]
                route_j = routes[rj]
                
                if (route_i[-1] == ni and route_j[0] == nj):
                    merged = route_i + route_j
                elif (route_i[-1] == ni and route_j[-1] == nj):
                    merged = route_i + route_j[::-1]
                elif (route_i[0] == ni and route_j[0] == nj):
                    merged = route_i[::-1] + route_j
                elif (route_i[0] == ni and route_j[-1] == nj):
                    merged = route_j + route_i
                else:
                    continue
                
                _, feasible, _ = self.evaluate_route_fast(merged, start_time)
                
                if feasible or len(routes) > 2:
                    routes[ri] = merged
                    del routes[rj]
                    for node in merged:
                        route_of[node] = ri
        
        best_route = min(routes.values(), key=lambda r: self.evaluate_route_fast(r, start_time)[0])
        return best_route
    
    def limited_2opt(self, route, start_time=0.0, max_iterations=100, max_no_improve=20):
        if len(route) <= 3:
            return route
        
        best_route = route[:]
        best_time, _, _ = self.evaluate_route_fast(best_route, start_time)
        
        iterations = 0
        no_improve_count = 0
        
        if self.n > 50:
            max_gap = 15
        elif self.n > 30:
            max_gap = 20
        else:
            max_gap = 30
        
        while iterations < max_iterations and no_improve_count < max_no_improve:
            improved = False
            iterations += 1
            
            for i in range(len(best_route) - 1):
                if improved:
                    break
                    
                for j in range(i + 2, min(i + max_gap, len(best_route))):
                    new_route = best_route[:i+1] + best_route[i+1:j+1][::-1] + best_route[j+1:]
                    
                    new_time, feasible, _ = self.evaluate_route_fast(new_route, start_time)
                    
                    if feasible and new_time < best_time:
                        best_route = new_route
                        best_time = new_time
                        improved = True
                        no_improve_count = 0
                        break
            
            if not improved:
                no_improve_count += 1
        
        return best_route
    
    def optimize_route(self, start_time=0.0, time_limit=4.8):
        opt_start = time.time()
        
        print(f"  Problem size: {self.n} nodes")
        
        if self.n <= 30:
            print("  Using: Nearest Neighbor + Savings")
            route1 = self.fast_nearest_neighbor(start_time)
            time1, feas1, _ = self.evaluate_route_fast(route1, start_time)
            
            route2 = self.savings_algorithm(start_time)
            time2, feas2, _ = self.evaluate_route_fast(route2, start_time)
            
            if feas1 and feas2:
                route = route1 if time1 < time2 else route2
            elif feas1:
                route = route1
            elif feas2:
                route = route2
            else:
                route = route1 if time1 < time2 else route2
        
        elif self.n <= 60:
            print("  Using: Savings Algorithm")
            route = self.savings_algorithm(start_time)
        
        else:
            print("  Using: Fast Nearest Neighbor")
            route = self.fast_nearest_neighbor(start_time)
        
        expected = set(self.points.keys()) - {self.depot_id}
        if set(route) != expected:
            missing = list(expected - set(route))
            route.extend(missing)
            print(f"  Added {len(missing)} missing nodes")
        
        total_time, feasible, arrival_times = self.evaluate_route_fast(route, start_time)
        print(f"  Initial: time={total_time:.1f}, feasible={feasible}")
        
        elapsed = time.time() - opt_start
        
        if elapsed < time_limit * 0.6 and feasible:
            if self.n > 70:
                max_iter = 50
            elif self.n > 50:
                max_iter = 100
            else:
                max_iter = 200
            
            print(f"  Running 2-opt (max {max_iter} iterations)...")
            route = self.limited_2opt(route, start_time, max_iterations=max_iter)
            total_time, feasible, arrival_times = self.evaluate_route_fast(route, start_time)
            print(f"  After 2-opt: time={total_time:.1f}, feasible={feasible}")
        
        computation_time = time.time() - opt_start
        
        result = {
            "route": [self.depot_id] + route,
            "delivery_sequence": route,
            "delivery_times": {},
            "time_windows": {},
            "total_time_minutes": round(total_time if total_time != float('inf') else -1, 2),
            "computation_time_seconds": round(computation_time, 4),
            "num_deliveries": len(route),
            "expected_deliveries": self.n,
            "all_nodes_visited": len(route) == self.n,
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


# ============================================================================
# JSON FUNCTIONS
# ============================================================================

def load_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    delivery_points = []
    for point in data.get('delivery_points', []):
        dp = DeliveryPoint(
            id=point['id'],
            x=point['x'],
            y=point['y'],
            earliest=point.get('earliest', 0),
            latest=point.get('latest', 10000),
            service_time=point.get('service_time', 5)
        )
        delivery_points.append(dp)
    
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


def save_result_to_json(result, output_file):
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

# checks if route is even possible 
#this is due to time windows being too tight becuase they are randomly generated
def create_feasible_input(filename, num_points=20):
    import random
    random.seed(42)
    
    city_size = 50
    base_speed = 40.0
    
    print(f"\nGenerating feasible instance with {num_points} nodes...")
    
    locations = [(0, 0.0, 0.0)]
    for i in range(1, num_points + 1):
        x = round(random.uniform(-city_size/2, city_size/2), 2)
        y = round(random.uniform(-city_size/2, city_size/2), 2)
        locations.append((i, x, y))
    
    unvisited = set(range(1, num_points + 1))
    sequence = [0]
    current = 0
    
    while unvisited:
        nearest = min(unvisited, key=lambda i: 
                     math.sqrt((locations[i][1] - locations[current][1])**2 + 
                              (locations[i][2] - locations[current][2])**2))
        sequence.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    delivery_points = []
    cumulative_time = 0
    
    for idx, node_id in enumerate(sequence):
        nid, x, y = locations[node_id]
        
        if idx > 0:
            prev_id = sequence[idx - 1]
            _, px, py = locations[prev_id]
            dist = math.sqrt((x - px)**2 + (y - py)**2)
            travel = (dist / base_speed) * 60
            cumulative_time += travel
        
        service = 10 if node_id > 0 else 0
        
        buffer = 100 + num_points
        earliest = max(0, cumulative_time - buffer)
        latest = cumulative_time + buffer + 500
        
        delivery_points.append({
            "id": nid,
            "x": x,
            "y": y,
            "earliest": round(earliest, 2),
            "latest": round(latest, 2),
            "service_time": service
        })
        
        cumulative_time += service
    
    traffic_conditions = []
    num_traffic = min(30, num_points)
    
    for _ in range(num_traffic):
        from_id = random.randint(0, num_points)
        to_id = random.randint(0, num_points)
        if from_id != to_id:
            traffic_conditions.append({
                "from_id": from_id,
                "to_id": to_id,
                "speed_factor": round(random.uniform(0.85, 1.15), 2),
                "delay": round(random.uniform(0, 2), 2)
            })
    
    data = {
        "delivery_points": delivery_points,
        "traffic_conditions": traffic_conditions,
        "config": {"depot_id": 0, "base_speed": 40.0, "start_time": 0}
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Created: {filename}")
    print(f"  Estimated min time: {cumulative_time:.1f} minutes")

# ============================================================================
# Output Functions
# ============================================================================
def print_results(result):
    print(f"\n{'='*60}")
    print("RESULTS")
    print('='*60)
    print(f"All nodes visited: {result['all_nodes_visited']}")
    print(f"Feasible: {result['feasible']}")
    print(f"Total time: {result['total_time_minutes']} minutes")
    print(f"Computation: {result['computation_time_seconds']} seconds")
    
    if result['computation_time_seconds'] < 5.0:
        print(f"Performance: PASS")
    else:
        print(f"Performance: FAIL")
    
    route_preview = result['route'][:15]
    print(f"\nRoute: {' → '.join(map(str, route_preview))} ... ({len(result['route'])-1} total)")
    
    violations = sum(1 for tw in result['time_windows'].values() if tw.get('violated', False))
    print(f"Time window violations: {violations}/{result['num_deliveries']}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("FAST ROUTE OPTIMIZATION SYSTEM")
    print("="*60)
    
    input_file = "traffic_data/delivery_100nodes.json"
    output_file = "output_data/optimized_route.json"
    
    try:
        with open(input_file, 'r') as f:
            pass
    except FileNotFoundError:
        num_points = int(input("\nEnter number of delivery points [100]: ") or "100")
        num_points = min(max(num_points, 5), 100)
        create_feasible_input(input_file, num_points)
    
    print(f"\nLoading: {input_file}")
    delivery_points, traffic_conditions, config = load_from_json(input_file)
    
    print(f"Loaded {len(delivery_points)} points ({len(delivery_points)-1} deliveries)")
    print(f"Traffic conditions: {len(traffic_conditions)}")
    
    print(f"\n{'='*60}")
    print("OPTIMIZING...")
    print('='*60)
    
    optimizer = FastRouteOptimizer(
        delivery_points=delivery_points,
        traffic_conditions=traffic_conditions,
        depot_id=config.get('depot_id', 0),
        base_speed=config.get('base_speed', 40.0)
    )
    
    start_time = config.get('start_time', 0)
    result = optimizer.optimize_route(start_time)
    
    print_results(result)
    
    save_result_to_json(result, output_file)
    print(f"\nSaved to: {output_file}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()