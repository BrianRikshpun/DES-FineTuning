"""
Bank Renege Simulation
Parameters:
  - new_customers: total number of customers arriving
  - interval_customers: mean inter-arrival time (exponential)
  - min_patience: minimum customer patience
  - max_patience: maximum customer patience
  - time_in_bank: mean service time (exponential)
  - num_counters: number of bank counters (capacity)

Output:
  - avg_wait_time: average wait time of served customers
  - renege_rate: fraction of customers who reneged
"""

import random
import simpy


def run(
    new_customers=5,
    interval_customers=10.0,
    min_patience=1.0,
    max_patience=3.0,
    time_in_bank=12.0,
    num_counters=1,
    seed=42,
):
    rng = random.Random(seed)
    wait_times = []
    reneged = [0]
    served = [0]

    def source(env, counter):
        for i in range(new_customers):
            env.process(customer(env, counter))
            t = rng.expovariate(1.0 / interval_customers)
            yield env.timeout(t)

    def customer(env, counter):
        arrive = env.now
        with counter.request() as req:
            patience = rng.uniform(min_patience, max_patience)
            results = yield req | env.timeout(patience)
            wait = env.now - arrive
            if req in results:
                served[0] += 1
                wait_times.append(wait)
                tib = rng.expovariate(1.0 / time_in_bank)
                yield env.timeout(tib)
            else:
                reneged[0] += 1

    env = simpy.Environment()
    counter = simpy.Resource(env, capacity=num_counters)
    env.process(source(env, counter))
    env.run()

    total = served[0] + reneged[0]
    avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0.0
    renege_rate = reneged[0] / total if total > 0 else 0.0
    return {
        "avg_wait_time": round(avg_wait, 4),
        "renege_rate": round(renege_rate, 4),
    }


PARAM_RANGES = {
    "new_customers": (10, 100),
    "interval_customers": (2.0, 20.0),
    "min_patience": (0.5, 5.0),
    "max_patience": (2.0, 15.0),
    "time_in_bank": (1.0, 30.0),
    "num_counters": (1, 5),
}

OUTPUT_KEY = "avg_wait_time"

QUESTION_TEMPLATE = (
    "In a bank simulation with {new_customers} customers arriving at a mean interval of "
    "{interval_customers:.1f} minutes, patience ranging from {min_patience:.1f} to "
    "{max_patience:.1f} minutes, mean service time of {time_in_bank:.1f} minutes, and "
    "{num_counters} counter(s), what is the average wait time of served customers (in minutes)?"
)
