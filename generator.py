# generator.py
import json, random, sys

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
random.seed(42)

data = {
    "delivery_points": [{"id": 0, "x": 0, "y": 0, "earliest": 0, "latest": 1440, "service_time": 0}] + 
                       [{"id": i, "x": round(random.uniform(-25, 25), 2), 
                         "y": round(random.uniform(-25, 25), 2),
                         "earliest": round(random.uniform(0, 300), 2),
                         "latest": round(random.uniform(400, 600), 2),
                         "service_time": 0} for i in range(1, n+1)],
    "traffic_conditions": [{"from_id": random.randint(0, n), "to_id": random.randint(0, n),
                           "speed_factor": round(random.uniform(0.6, 1.4), 2), "delay": 0}
                          for _ in range(n*2)],
    "config": {"depot_id": 0, "base_speed": 40, "start_time": 0}
}

with open(f"traffic_data/delivery_{n}nodes.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Generated delivery_{n}nodes.json with {n} delivery points within the traffic_data folder.")