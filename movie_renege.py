"""
Movie Renege Simulation
Parameters:
  - num_movies: number of movies showing
  - tickets_per_movie: initial tickets for each movie
  - sellout_threshold: trigger sellout when tickets below this
  - arrival_rate: mean inter-arrival time (minutes, exponential)
  - max_tickets_per_buyer: max tickets one person buys (uniform 1..N)
  - sim_time: simulation duration (minutes)

Output:
  - avg_renegers_per_movie: average number of renegers per movie
  - pct_sold_out: percentage of movies that sold out
"""

import random
import simpy


def run(
    num_movies=3,
    tickets_per_movie=50,
    sellout_threshold=2,
    arrival_rate=0.5,
    max_tickets_per_buyer=6,
    sim_time=120.0,
    seed=42,
):
    rng = random.Random(seed)
    movies = [f"Movie_{i}" for i in range(num_movies)]
    available = {m: tickets_per_movie for m in movies}
    num_renegers = {m: 0 for m in movies}
    sold_out_time = {m: None for m in movies}

    def moviegoer(env, movie, num_tickets, counter, sold_out_events):
        with counter.request() as my_turn:
            result = yield my_turn | sold_out_events[movie]
            if my_turn not in result:
                num_renegers[movie] += 1
                return
            if available[movie] < num_tickets:
                yield env.timeout(0.5)
                return
            available[movie] -= num_tickets
            if available[movie] < sellout_threshold:
                if not sold_out_events[movie].triggered:
                    sold_out_events[movie].succeed()
                    sold_out_time[movie] = env.now
                    available[movie] = 0
            yield env.timeout(1)

    def customer_arrivals(env, counter, sold_out_events):
        while True:
            yield env.timeout(rng.expovariate(1.0 / arrival_rate))
            movie = rng.choice(movies)
            num_tickets = rng.randint(1, max_tickets_per_buyer)
            if available[movie] > 0:
                env.process(moviegoer(env, movie, num_tickets, counter, sold_out_events))

    env = simpy.Environment()
    counter = simpy.Resource(env, capacity=1)
    sold_out_events = {m: env.event() for m in movies}
    env.process(customer_arrivals(env, counter, sold_out_events))
    env.run(until=sim_time)

    total_renegers = sum(num_renegers.values())
    avg_renegers = total_renegers / num_movies if num_movies > 0 else 0.0
    sold_out_count = sum(1 for t in sold_out_time.values() if t is not None)
    pct_sold_out = sold_out_count / num_movies * 100.0

    return {
        "avg_renegers_per_movie": round(avg_renegers, 4),
        "pct_sold_out": round(pct_sold_out, 4),
    }


PARAM_RANGES = {
    "num_movies": (2, 8),
    "tickets_per_movie": (20, 200),
    "sellout_threshold": (1, 5),
    "arrival_rate": (0.2, 2.0),
    "max_tickets_per_buyer": (1, 10),
    "sim_time": (60.0, 360.0),
}

OUTPUT_KEY = "avg_renegers_per_movie"

QUESTION_TEMPLATE = (
    "In a movie theater simulation with {num_movies} movies, {tickets_per_movie} tickets each, "
    "sellout threshold {sellout_threshold}, customers arriving every {arrival_rate:.2f} min on "
    "average, buying up to {max_tickets_per_buyer} tickets, running for {sim_time:.0f} min, "
    "what is the average number of people who reneged (left the queue) per movie?"
)
