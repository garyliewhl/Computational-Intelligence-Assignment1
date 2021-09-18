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

        # set position to the min or max value if it goes out of bounds
        if self.position > position_limits[1]:
            self.position = position_limits[1]
        elif self.position < position_limits[0]:
            self.position = position_limits[0]
    
    def update_velocity(self, inertia, alpha, beta, global_best):
        self.velocity = (inertia * self.velocity) + (alpha[0] * beta[0] * (self.local_best - self.position)) + (alpha[1] * beta[1] * (global_best - self.position))

    def update_local_best(self):
        if calc_fitness(self.position) > calc_fitness(self.local_best):
            self.local_best = self.position

def calc_acc_cost(position):
    days = position // 1440
    return round((30 + (days * 30) + ((7 - days) * 25)), 2)

def calc_ren_lvl(position):
    T = position / 1440
    return ((T**2) / 126) + (T / 63) + 0.5

def calc_mov_cost(position):
    t = (position % 1440) / 60
    return round(((50 * math.cos((12 * math.pi * t) / 24)) + (50 * math.cos((8 * math.pi * t) / 24)) + 150),2)

def calc_fitness(position):
    # fixed values for the minimum and maximum values of each variable
    ren_lvl_range = [0.5, 1]
    acc_cost_range = [205, 235]
    mov_cost_range = [68.29, 250]

    # calculate the cost and levels
    accommodation_cost = calc_acc_cost(position)
    renovation_level = calc_ren_lvl(position)
    moving_cost = calc_mov_cost(position)

    ### using cost and levels to calculate fitness

    ## fitness function 1, all values are unscaled, left to default ranges
    # fitness = renovation_level
    # fitness -= (accommodation_cost + moving_cost)

    ## fitness function 2, all values are scaled to 0-1
    # fitness = (renovation_level - ren_lvl_range[0]) / (ren_lvl_range[1] - ren_lvl_range[0])
    # fitness += 1 - (((accommodation_cost - acc_cost_range[0]) + (moving_cost - mov_cost_range[0])) / ((acc_cost_range[1] - acc_cost_range[0]) + (mov_cost_range[1] - mov_cost_range[0])))

    ## fitness function 3, renovation level scaled to range of 0-1 then multipled by the range of cost.
    # fitness = (renovation_level - ren_lvl_range[0]) / (ren_lvl_range[1] - ren_lvl_range[0]) 
    # fitness *= ((mov_cost_range[1] - mov_cost_range[0]) + (acc_cost_range[1] - acc_cost_range[0]))
    # fitness -= (accommodation_cost + moving_cost)

    ### fitness function 4, slight modification of fitness function 3.
    fitness = renovation_level * ((mov_cost_range[1] - mov_cost_range[0]) + (acc_cost_range[1] - acc_cost_range[0]))
    fitness -= (accommodation_cost + moving_cost)

    return fitness

def check_global_best(local_best, global_best):
    if calc_fitness(local_best) > calc_fitness(global_best):
        return local_best
    else:
        return global_best


def calc_avg_fit_diff(particles):
    fitness = list(map(calc_fitness, [particle.position for particle in particles]))
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
    particles = []
    for num in range(num_particles):
        particles.append(Particle(random.randint(position_limits[0], position_limits[1])))
    return particles

def main():
    # fixed variables
    position_cap = 10079 # minute representation of Sunday 11:59

    # parameters initialization
    alpha = [0.01, 0.1]
    inertia_weight = 0.5
    num_particles = 50
    global_best = None
    position_limits = [0, position_cap]
    curr_iter = 0

    # termination condition
    max_iter = 2000
    min_avg_fit_diff = 0.01
    min_avg_dis_diff = 0.01

    particles = initialize_particles(num_particles, position_limits)
    global_best = particles[0].position
    while((curr_iter < max_iter) and (calc_avg_fit_diff(particles) > min_avg_fit_diff) and (calc_avg_pos_diff(particles) > min_avg_dis_diff)):
        print("Iteration:", curr_iter)
        print("Particles:", [particle.position for particle in particles])
        print("Fitness:", [calc_fitness(particle.position) for particle in particles])
        print("Global best:", global_best, calc_fitness(global_best))

        if curr_iter % (max_iter//10) == 0:
                    inertia_weight -= 0.1

        for particle in particles:
            particle.update_local_best()
            global_best = check_global_best(particle.local_best, global_best)
            
        beta = [random.random(), random.random()]

        for particle in particles:
            particle.update_velocity(inertia_weight, alpha, beta, global_best)
            particle.update_position(position_limits)
        
        curr_iter += 1

    print("Final Results:")
    print("Iteration:", curr_iter)
    print("Particles:", [particle.position for particle in particles])
    print("Fitness:", [calc_fitness(particle.position) for particle in particles])
    print("Global best:", global_best, calc_fitness(global_best))
    print("Cost:", (calc_acc_cost(global_best) + calc_mov_cost(global_best)), "Renovation:", calc_ren_lvl(global_best))

if __name__ == "__main__":
    main()