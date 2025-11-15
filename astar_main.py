import json
import numpy as np
import heapq
import time
from typing import Dict

class RouteOptimizer:
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.points = data['delivery_points']
        self.n = len(self.points)
        
        if self.n < 2:
            raise Exception("Need at least 2 points (depot + 1 delivery)")
        
        self.coords = np.array([[p['x'], p['y']] for p in self.points], dtype=np.float32)
        self.earliest = np.array([p['earliest'] for p in self.points], dtype=np.float32)
        self.latest = np.array([p['latest'] for p in self.points], dtype=np.float32)
        self.service_time = np.array([p.get('service_time', 0) for p in self.points], dtype=np.float32)
        
        self.speed = data.get('base_speed', 40.0)
        self.lateness_penalty = 100.0
        
        diff = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
        self.dist = np.sqrt(np.sum(diff ** 2, axis=2))
        self.time_matrix = (self.dist / self.speed) * 60
        
        if self.n <= 20:
            self.beam_width = 100000
        elif self.n <= 50:
            self.beam_width = 5000
        else:
            self.beam_width = 1000
    
    # ============ CALCULATION FUNCTIONS ============
    
    def get_arrival(self, curr_time, from_id, to_id):
        depart = curr_time + self.service_time[from_id]
        arrive = depart + self.time_matrix[from_id, to_id]
        return max(arrive, self.earliest[to_id])
    
    def calc_lateness_penalty(self, arrival_time, node_id):
        if arrival_time <= self.latest[node_id]:
            return 0.0
        lateness = arrival_time - self.latest[node_id]
        return lateness * self.lateness_penalty
    
    def heuristic(self, node, unvisited):
        if not unvisited:
            return self.dist[node, 0]
        
        uv = list(unvisited)
        uv_arr = np.array(uv, dtype=np.int32)
        
        min_reach = np.min(self.dist[node, uv_arr])
        
        mst = 0.0
        if len(uv) > 1:
            uv_dists = self.dist[np.ix_(uv_arr, uv_arr)]
            np.fill_diagonal(uv_dists, np.inf)
            mst = np.sum(np.min(uv_dists, axis=1)) / 2
        
        min_back = np.min(self.dist[uv_arr, 0])
        
        return min_reach + mst + min_back
    
    def calc_times(self, route):
        times = np.zeros(len(route), dtype=np.float32)
        t = 0.0
        
        for i in range(len(route) - 1):
            t = self.get_arrival(t, route[i], route[i+1])
            times[i+1] = t
        
        return times
    
    def total_dist(self, route):
        d = 0.0
        for i in range(len(route)-1):
            d += self.dist[route[i], route[i+1]]
        return d
    
    def greedy_solution(self):
        route = [0]
        unvisited = set(range(1, self.n))
        curr_time = 0.0
        curr = 0
        
        while unvisited:
            best = None
            best_score = float('inf')
            
            for nxt in unvisited:
                arr = self.get_arrival(curr_time, curr, nxt)
                
                dist = self.dist[curr, nxt]
                penalty = self.calc_lateness_penalty(arr, nxt)
                
                remaining = unvisited - {nxt}
                h = self.heuristic(nxt, frozenset(remaining)) if remaining else self.dist[nxt, 0]
                
                score = dist + penalty + h * 0.5
                
                if score < best_score:
                    best_score = score
                    best = nxt
            
            if best is None:
                best = list(unvisited)[0]
            
            route.append(best)
            unvisited.remove(best)
            curr_time = self.get_arrival(curr_time, curr, best)
            curr = best
        
        route.append(0)
        return route
    
    def solve(self):
        t0 = time.time()
        
        greedy_route = self.greedy_solution()
        greedy_times = self.calc_times(greedy_route)
        greedy_dist = self.total_dist(greedy_route)
        
        greedy_penalty = 0.0
        for i in range(len(greedy_route)):
            greedy_penalty += self.calc_lateness_penalty(greedy_times[i], greedy_route[i])
        
        greedy_cost = greedy_dist + greedy_penalty
        
        cnt = 0
        initial = {
            'g': 0.0,
            'penalty': 0.0,
            'time': 0.0,
            'node': 0,
            'visited': frozenset([0]),
            'path': [0]
        }
        
        h = self.heuristic(0, frozenset(range(1, self.n)))
        heap = [(h, cnt, initial)]
        cnt += 1
        
        seen = {}
        seen[(0, frozenset([0]))] = 0.0
        
        explored = 0
        best_route = greedy_route
        best_cost = greedy_cost
        best_times = greedy_times
        
        while heap and time.time() - t0 < 4.0:
            f, _, state = heapq.heappop(heap)
            
            g = state['g']
            penalty = state['penalty']
            curr_time = state['time']
            curr = state['node']
            visited = state['visited']
            path = state['path']
            
            explored += 1
            
            if f >= best_cost:
                continue
            
            if len(visited) == self.n:
                ret_time = self.get_arrival(curr_time, curr, 0)
                ret_penalty = self.calc_lateness_penalty(ret_time, 0)
                
                total_cost = g + self.dist[curr, 0] + penalty + ret_penalty
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    complete_route = path + [0]
                    best_route = complete_route
                    best_times = self.calc_times(complete_route)
                
                continue
            
            unvisited = set(range(self.n)) - visited
            
            neighbors = []
            for nxt in unvisited:
                arr = self.get_arrival(curr_time, curr, nxt)
                
                edge_distance = self.dist[curr, nxt]
                new_g = g + edge_distance
                
                late_penalty = self.calc_lateness_penalty(arr, nxt)
                new_penalty = penalty + late_penalty
                
                new_visited = visited | {nxt}
                new_path = path + [nxt]
                
                key = (nxt, new_visited)
                
                total_cost_to_state = new_g + new_penalty
                if key in seen and seen[key] <= total_cost_to_state:
                    continue
                
                seen[key] = total_cost_to_state
                
                remaining = unvisited - {nxt}
                h_val = self.heuristic(nxt, frozenset(remaining))
                
                new_f = new_g + new_penalty + h_val
                
                new_state = {
                    'g': new_g,
                    'penalty': new_penalty,
                    'time': arr,
                    'node': nxt,
                    'visited': new_visited,
                    'path': new_path
                }
                
                neighbors.append((new_f, cnt, new_state))
                cnt += 1
            
            for item in neighbors:
                heapq.heappush(heap, item)
            
            if len(heap) > self.beam_width:
                heap = heapq.nsmallest(self.beam_width, heap)
                heapq.heapify(heap)
        
        elapsed = time.time() - t0
        
        return best_route, best_times, elapsed, explored
    
    # ============ OUTPUT FUNCTIONS ============
    
    def make_output(self, route, times):
        deliveries = []
        late_count = 0
        total_lateness = 0.0
        
        for i in range(len(route)):
            nid = route[i]
            arr = times[i]
            early = self.earliest[nid]
            late = self.latest[nid]
            wait = max(0, early - arr)
            lateness = max(0, arr - late)
            ok = arr <= late
            
            if not ok:
                late_count += 1
                total_lateness += lateness
            
            deliveries.append({
                "node_id": int(nid),
                "arrival_time": float(arr),
                "earliest": float(early),
                "latest": float(late),
                "waiting_time": float(wait),
                "lateness": float(lateness),
                "on_time": ok
            })
        
        return {
            "route": route,
            "delivery_times": deliveries,
            "total_distance": float(self.total_dist(route)),
            "total_time": float(times[-1]) if len(times) > 0 else 0.0,
            "num_deliveries": len(route) - 2,
            "late_deliveries": late_count,
            "total_lateness": float(total_lateness)
        }


def save_result_to_json(result: Dict, output_file: str):
    """Save results to JSON"""
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)


def main():
    import sys
    
    infile = sys.argv[1] if len(sys.argv) > 1 else 'traffic_data/delivery_100nodes.json'
    outfile = sys.argv[2] if len(sys.argv) > 2 else 'route_output.json'
    
    try:
        print(f"Loading {infile}...")
        opt = RouteOptimizer(infile)
        print(f"Solving for {opt.n-1} delivery points...")
        
        route, times, elapsed, explored = opt.solve()
        output = opt.make_output(route, times)
        
        print(f"\nSolved in {elapsed:.2f} seconds")
        print(f"Total distance: {output['total_distance']:.2f}")
        print(f"Late deliveries: {output['late_deliveries']}/{output['num_deliveries']}")
        
        #save_result_to_json(output, outfile)
        
    except FileNotFoundError:
        print(f"Error: File '{infile}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()



