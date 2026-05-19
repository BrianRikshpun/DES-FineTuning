"""
Carwash Simulation
Parameters:
  - num_machines: number of carwash machines
  - mean_washtime: mean wash duration (minutes)
  - arrival_interval: mean inter-arrival time (minutes)
  - sim_time: total simulation time (minutes)

Output:
  - avg_wait_time: average waiting time before wash starts
  - cars_washed: total number of cars washed
"""

import random
import simpy


def run(
    num_machines=2,
    mean_washtime=5.0,
    arrival_interval=3.5,
    sim_time=120.0,
    seed=42,
):
    rng = random.Random(seed)
    wait_times = []

    def car(name, env, carwash):
        arrive = env.now
        with carwash.request() as req:
            yield req
            wait = env.now - arrive
            wait_times.append(wait)
            wash_time = rng.expovariate(1.0 / mean_washtime)
            yield env.timeout(wash_time)

    def car_generator(env, carwash):
        i = 0
        while True:
            yield env.timeout(rng.expovariate(1.0 / arrival_interval))
            env.process(car(f"Car {i}", env, carwash))
            i += 1

    env = simpy.Environment()
    carwash = simpy.Resource(env, capacity=num_machines)
    env.process(car_generator(env, carwash))
    env.run(until=sim_time)

    avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0.0
    return {
        "avg_wait_time": round(avg_wait, 4),
        "cars_washed": len(wait_times),
    }


PARAM_RANGES = {
    "num_machines": (1, 6),
    "mean_washtime": (2.0, 15.0),
    "arrival_interval": (1.0, 10.0),
    "sim_time": (60.0, 480.0),
}

OUTPUT_KEY = "avg_wait_time"

QUESTION_TEMPLATE = (
    "In a carwash simulation with {num_machines} machine(s), mean wash time of {mean_washtime:.1f} "
    "minutes, cars arriving every {arrival_interval:.1f} minutes on average, run for {sim_time:.0f} "
    "minutes, what is the average customer wait time before washing begins (in minutes)?"
)
