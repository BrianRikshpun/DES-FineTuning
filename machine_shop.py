"""
Machine Shop Simulation
Parameters:
  - num_machines: number of machines
  - pt_mean: mean processing time per part (minutes)
  - pt_sigma: sigma of processing time
  - mttf: mean time to failure (minutes)
  - repair_time: time to repair a machine (minutes)
  - weeks: simulation duration in weeks (capped low for speed)

Output:
  - avg_parts_made: average parts made per machine
  - total_parts: total parts made across all machines
"""

import random
import simpy


def run(
    num_machines=10,
    pt_mean=10.0,
    pt_sigma=2.0,
    mttf=300.0,
    repair_time=30.0,
    weeks=2,
    seed=42,
):
    rng = random.Random(seed)
    sim_time = weeks * 7 * 24 * 60
    break_mean = 1.0 / mttf

    def time_per_part():
        t = rng.gauss(pt_mean, pt_sigma)
        while t <= 0:
            t = rng.gauss(pt_mean, pt_sigma)
        return t

    def time_to_failure():
        return rng.expovariate(break_mean)

    class Machine:
        def __init__(self, env, name, repairman):
            self.env = env
            self.name = name
            self.parts_made = 0
            self.broken = False
            self.process = env.process(self.working(repairman))
            env.process(self.break_machine())

        def working(self, repairman):
            while True:
                done_in = time_per_part()
                while done_in:
                    start = self.env.now
                    try:
                        yield self.env.timeout(done_in)
                        done_in = 0
                    except simpy.Interrupt:
                        self.broken = True
                        done_in -= self.env.now - start
                        with repairman.request(priority=1) as req:
                            yield req
                            yield self.env.timeout(repair_time)
                        self.broken = False
                self.parts_made += 1

        def break_machine(self):
            while True:
                yield self.env.timeout(time_to_failure())
                if not self.broken:
                    self.process.interrupt()

    def other_jobs(env, repairman):
        while True:
            done_in = 30.0
            while done_in:
                with repairman.request(priority=2) as req:
                    yield req
                    start = env.now
                    try:
                        yield env.timeout(done_in)
                        done_in = 0
                    except simpy.Interrupt:
                        done_in -= env.now - start

    env = simpy.Environment()
    repairman = simpy.PreemptiveResource(env, capacity=1)
    machines = [Machine(env, f"Machine {i}", repairman) for i in range(num_machines)]
    env.process(other_jobs(env, repairman))
    env.run(until=sim_time)

    parts = [m.parts_made for m in machines]
    avg_parts = sum(parts) / len(parts) if parts else 0.0
    return {
        "avg_parts_made": round(avg_parts, 4),
        "total_parts": sum(parts),
    }


PARAM_RANGES = {
    "num_machines": (2, 15),
    "pt_mean": (5.0, 20.0),
    "pt_sigma": (0.5, 4.0),
    "mttf": (150.0, 600.0),
    "repair_time": (10.0, 60.0),
    "weeks": (1, 2),       # kept small for generation speed
}

OUTPUT_KEY = "avg_parts_made"

QUESTION_TEMPLATE = (
    "In a machine shop simulation with {num_machines} machines, mean processing time of "
    "{pt_mean:.1f} min (sigma {pt_sigma:.1f}), mean time to failure of {mttf:.0f} min, "
    "repair time of {repair_time:.0f} min, running for {weeks} week(s), what is the average "
    "number of parts made per machine?"
)
