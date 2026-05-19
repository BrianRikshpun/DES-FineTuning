"""
Gas Station Refueling Simulation
Parameters:
  - num_pumps: number of fuel pumps
  - tank_size: gas station tank capacity (liters)
  - threshold_pct: refuel threshold (% of tank)
  - car_tank_size: car tank capacity (liters)
  - refueling_speed: liters per second
  - tank_truck_time: time for truck to arrive (seconds)
  - t_inter_min / t_inter_max: car arrival interval range (seconds)
  - sim_time: simulation duration (seconds)

Output:
  - cars_served: total cars that completed refueling
  - avg_refuel_amount: average fuel per car (liters)
"""

import itertools
import random
import simpy


def run(
    num_pumps=2,
    tank_size=200.0,
    threshold_pct=25.0,
    car_tank_size=50.0,
    refueling_speed=2.0,
    tank_truck_time=300.0,
    t_inter_min=30,
    t_inter_max=300,
    sim_time=1000.0,
    seed=42,
):
    rng = random.Random(seed)
    cars_served = [0]
    refuel_amounts = []

    def car(name, env, gas_station, station_tank):
        car_level = rng.randint(5, int(car_tank_size * 0.5))
        with gas_station.request() as req:
            yield req
            fuel_needed = car_tank_size - car_level
            yield station_tank.get(fuel_needed)
            yield env.timeout(fuel_needed / refueling_speed)
            cars_served[0] += 1
            refuel_amounts.append(fuel_needed)

    def gas_station_control(env, station_tank):
        while True:
            if station_tank.level / station_tank.capacity * 100 < threshold_pct:
                yield env.process(tank_truck(env, station_tank))
            yield env.timeout(10)

    def tank_truck(env, station_tank):
        yield env.timeout(tank_truck_time)
        amount = station_tank.capacity - station_tank.level
        yield station_tank.put(amount)

    def car_generator(env, gas_station, station_tank):
        for i in itertools.count():
            yield env.timeout(rng.randint(t_inter_min, t_inter_max))
            env.process(car(f"Car {i}", env, gas_station, station_tank))

    env = simpy.Environment()
    gas_station = simpy.Resource(env, capacity=num_pumps)
    station_tank = simpy.Container(env, tank_size, init=tank_size)
    env.process(gas_station_control(env, station_tank))
    env.process(car_generator(env, gas_station, station_tank))
    env.run(until=sim_time)

    avg_refuel = sum(refuel_amounts) / len(refuel_amounts) if refuel_amounts else 0.0
    return {
        "cars_served": cars_served[0],
        "avg_refuel_amount": round(avg_refuel, 4),
    }


PARAM_RANGES = {
    "num_pumps": (1, 6),
    "tank_size": (100.0, 1000.0),
    "threshold_pct": (10.0, 40.0),
    "car_tank_size": (30.0, 80.0),
    "refueling_speed": (1.0, 5.0),
    "tank_truck_time": (100.0, 600.0),
    "t_inter_min": (10, 60),
    "t_inter_max": (100, 500),
    "sim_time": (500.0, 5000.0),
}

OUTPUT_KEY = "cars_served"

QUESTION_TEMPLATE = (
    "In a gas station simulation with {num_pumps} pump(s), tank capacity {tank_size:.0f}L, "
    "refill threshold {threshold_pct:.0f}%, car tank size {car_tank_size:.0f}L, refueling "
    "speed {refueling_speed:.1f} L/s, truck arrival time {tank_truck_time:.0f}s, car arrival "
    "interval [{t_inter_min}s, {t_inter_max}s], running for {sim_time:.0f}s, how many cars "
    "were fully served?"
)
