import numpy as np
import random

# Particle Swarm Optimization Parameters
num_particles = 30
max_iterations = 100
inertia_weight = 0.7
c1, c2 = 1.5, 1.5  # Cognitive and social coefficients

# Evaluate the travel time based on signal timings
def evaluate_signal_timing(signal_timings, traffic_flows, road_lengths):
    total_travel_time = 0
    for i in range(len(signal_timings)):
        green_time = signal_timings[i]
        flow_rate = traffic_flows[i]
        road_length = road_lengths[i]

        # Travel time = (Road length) / (Flow rate × Green time)
        travel_time = road_length / (flow_rate * green_time + 1e-5)  # Add small value to avoid division by zero
        total_travel_time += travel_time
    return total_travel_time

# Initialize particles
def initialize_particles(num_intersections):
    particles = []
    for _ in range(num_particles):
        particle = {
            "position": np.random.uniform(10, 60, num_intersections),  # Random green times (10 to 60 seconds)
            "velocity": np.random.uniform(-5, 5, num_intersections),
            "best_position": None,
            "best_score": float("inf"),
        }
        particle["best_position"] = particle["position"].copy()
        particles.append(particle)
    return particles

# Update particle positions and velocities
def update_particles(particles, global_best, inertia_weight, c1, c2):
    for particle in particles:
        r1, r2 = np.random.random(), np.random.random()
        cognitive = c1 * r1 * (particle["best_position"] - particle["position"])
        social = c2 * r2 * (global_best["position"] - particle["position"])
        particle["velocity"] = inertia_weight * particle["velocity"] + cognitive + social

        # Update position and ensure it's within bounds (10 to 60 seconds)
        particle["position"] += particle["velocity"]
        particle["position"] = np.clip(particle["position"], 10, 60)

# PSO Algorithm for Traffic Flow Optimization
def pso_traffic_flow(traffic_flows, road_lengths):
    num_intersections = len(traffic_flows)
    particles = initialize_particles(num_intersections)
    global_best = {"position": None, "score": float("inf")}

    for iteration in range(max_iterations):
        for particle in particles:
            score = evaluate_signal_timing(particle["position"], traffic_flows, road_lengths)
            if score < particle["best_score"]:
                particle["best_score"] = score
                particle["best_position"] = particle["position"].copy()

            if score < global_best["score"]:
                global_best["score"] = score
                global_best["position"] = particle["position"].copy()

        update_particles(particles, global_best, inertia_weight, c1, c2)
        print(f"Iteration {iteration + 1}/{max_iterations} - Best Travel Time: {global_best['score']:.2f}")

    return global_best

# Main Function
if __name__ == "__main__":
    print("Traffic Flow Optimization using Particle Swarm Optimization")
    num_intersections = int(input("Enter the number of intersections: "))

    print("\nEnter traffic flow rates (vehicles per second) for each intersection:")
    traffic_flows = [float(input(f"Flow rate for Intersection {i + 1}: ")) for i in range(num_intersections)]

    print("\nEnter road lengths (meters) leading to each intersection:")
    road_lengths = [float(input(f"Road length for Intersection {i + 1}: ")) for i in range(num_intersections)]

    print("\nStarting optimization process...")
    best_solution = pso_traffic_flow(traffic_flows, road_lengths)

    print("\nOptimized Signal Timings (Green Light Durations in Seconds):")
    for i, timing in enumerate(best_solution["position"], start=1):
        print(f"Intersection {i}: {timing:.2f} seconds")

    print("Total Travel Time:", best_solution["score"])
