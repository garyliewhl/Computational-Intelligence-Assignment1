import numpy as np
import random
import math

class Particle:


    
    def __init__(self, position, velocity = 0):
        self.position = position
        self.velocity = velocity
        self.local_best = position
    
    def update_position(self, position_limits):
        self.position = round((self.position + self.velocity), 0)
        if self.position > position_limits[1]:
            self.position = position_limits[1]
        elif self.position < position_limits[0]:
            self.position = position_limits[0]
    
    def update_velocity(self, inertia, alpha, beta, global_best):
        self.velocity = (inertia * self.velocity) + (alpha[0] * beta[0] * (self.local_best - self.position)) + (alpha[1] * beta[1] * (global_best - self.position))

    def update_local_best(self):
        if calculate_fitness(self.position) > calculate_fitness(self.local_best):
            self.local_best = self.position
            check_global_best(self.local_best)


def calculate_fitness(position):
    global minimum_rennovation
    global maximum_rennovation
    global minimum_accomodation_cost
    global maximum_accomodation_cost
    global minimum_moving_cost
    global maximum_moving_cost

    minutes = position

    days = minutes // 1440
    minutes %= 1440
    hours = minutes // 60
    minutes %= 60

    accomodation_cost = round((30 + (days * 30) + ((7 - days) * 25)), 2)

    T = days + (((60 * hours) + minutes) / (1440))
    renovation_level = ((T**2) / 126) + (T / 63) + 0.5

    t = hours + (minutes / 60)
    moving_cost = round(((50 * math.cos((12 * math.pi * t) / 24)) + (50 * math.cos((8 * math.pi * t) / 24)) + 150),2)

    fitness = (renovation_level - minimum_rennovation) / (maximum_rennovation - minimum_rennovation)
    # fitness -= (accomodation_cost + moving_cost)

    fitness += 1 - ((accomodation_cost - minimum_accomodation_cost) / (maximum_accomodation_cost - minimum_accomodation_cost))
    fitness += 1 - ((moving_cost - minimum_moving_cost) / (maximum_moving_cost - minimum_moving_cost))

    return fitness

def check_global_best(local_best):
    global global_best
    
    if calculate_fitness(local_best) > calculate_fitness(global_best):
        print("Before:", calculate_fitness(global_best), global_best)
        global_best = local_best
        print("After:", calculate_fitness(global_best), global_best)

def calc_avg_fit_diff(particles):
    fitness = list(map(calculate_fitness, [particle.position for particle in particles]))
    mean_fitness = sum(fitness) / len(fitness)
    avg_fit_diff = 0
    for fit in fitness:
        avg_fit_diff += abs(fit - mean_fitness)
    avg_fit_diff /= len(fitness)
    return avg_fit_diff

def calc_avg_pos_diff(particles):
    mean_poition = sum([particle.position for particle in particles]) / len(particles)
    avg_pos_diff = 0
    for particle in particles:
        avg_pos_diff += abs(particle.position - mean_poition)
    avg_pos_diff /= len(particles)
    return avg_pos_diff

def initialize_particles(num_particles, position_limits):
    global global_best
    particles = []
    for num in range(num_particles):
        particles.append(Particle(random.randint(position_limits[0], position_limits[1])))
    global_best = particles[0].position
    return particles

def main():
    # fixed variables (no touchy touchy)
    global binary_length
    binary_length = 14
    global maximum_minute
    maximum_minute = 10079

    global minimum_rennovation
    minimum_rennovation = 0.5
    global maximum_rennovation
    maximum_rennovation = 1
    global minimum_accomodation_cost
    minimum_accomodation_cost = 205
    global maximum_accomodation_cost
    maximum_accomodation_cost = 235
    global minimum_moving_cost
    minimum_moving_cost = 68.29
    global maximum_moving_cost
    maximum_moving_cost = 250

    # parameters initialization
    alpha = [0.01, 0.1]
    inertia_weight = 0.5
    num_particles = 50
    global global_best
    global_best = None
    position_limits = [0, maximum_minute]
    curr_iter = 0

    # termination condition
    max_iter = 2000
    min_avg_fit_diff = 0
    min_avg_dis_diff = 0

    particles = initialize_particles(num_particles, position_limits)
    while((curr_iter <= max_iter) and (calc_avg_fit_diff(particles) > min_avg_fit_diff) and (calc_avg_pos_diff(particles) > min_avg_dis_diff)):
        print("Iteration:", curr_iter)
        print("Particles:", [particle.position for particle in particles])
        print("Fitness:", [calculate_fitness(particle.position) for particle in particles])

        for particle in particles:
            particle.update_local_best()
            
        beta = [random.random(), random.random()]

        for particle in particles:
            particle.update_velocity(inertia_weight, alpha, beta, global_best)
            particle.update_position(position_limits)
        
        curr_iter += 1
    print(global_best, calculate_fitness(global_best))

if __name__ == "__main__":
    main()