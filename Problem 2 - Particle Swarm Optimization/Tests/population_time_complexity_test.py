import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time

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

    ## fitness function, all values are scaled to 0-1
    fitness = renovation_level
    fitness += 1 - (((accommodation_cost - acc_cost_range[0]) + (moving_cost - mov_cost_range[0])) / ((acc_cost_range[1] - acc_cost_range[0]) + (mov_cost_range[1] - mov_cost_range[0])))

    return fitness

def check_global_best(local_best, global_best):
    if calc_fitness(local_best) > calc_fitness(global_best):
        return local_best
    else:
        return global_best


def calc_avg_fit_diff(particles):
    fitness = list(map(calc_fitness, [particle.position for particle in particles]))
    mean_fitness = sum(fitness) / len(fitness)
    difference = [abs(fit-mean_fitness) for fit in fitness]
    avg_fit_diff = sum(difference)/len(difference)
    return avg_fit_diff
    return avg_fit_diff

def calc_avg_pos_diff(particles):
    position_list = [particle.position for particle in particles]
    mean_position = sum(position_list) / len(position_list)
    difference = [abs(pos - mean_position) for pos in position_list]
    avg_pos_diff = sum(difference)/len(difference)
    return avg_pos_diff

def initialize_particles(num_particles, position_limits, spawn_segments):
    particles = []
    pop_segment = num_particles//spawn_segments
    range_val = position_limits[1]//spawn_segments
    min_val = position_limits[0]
    max_val=range_val
    for segment in range(spawn_segments):
        for num in range(pop_segment):
            particles.append(Particle(random.randint(min_val, max_val)))
        min_val+=range_val
        max_val+=range_val
    return particles

def main():
    computation_list = []
    
    for i in range(1, 101):
        random.seed(5)
        # fixed variables
        position_cap = 10079 # minute representation of Sunday 11:59

        # parameters 
        alpha = [0.5, 0.5]
        alpha_change = 0.1 # How much alpha values increase or decrease at split
        inertia_weight = 1
        inertia_change = 0.2 # How much inertia weight decrease at split
        iteration_split = 5 # Iteration split - alpha and inertia values increase/decrease at split
        num_particles = 10*i
        spawn_segments = 10 # Initialization splits
        
        # termination condition
        max_iter = 100
        min_avg_fit_diff = 0.01
        min_avg_dis_diff = 0.01

        # initialization
        global_best = None
        position_limits = [0, position_cap]
        curr_iter = 0

        particles = initialize_particles(num_particles, position_limits, spawn_segments)
        global_best = particles[0].position
        start = time.time()
        while((curr_iter < max_iter) and (calc_avg_fit_diff(particles) > min_avg_fit_diff) and (calc_avg_pos_diff(particles) > min_avg_dis_diff)):
            
            if (curr_iter % (max_iter//iteration_split) == 0) and (curr_iter != 0):
                inertia_weight -= inertia_change
                alpha[0] -= alpha_change
                alpha[1] += alpha_change

            for particle in particles:
                particle.update_local_best()
                global_best = check_global_best(particle.local_best, global_best)
                
            beta = [1,1]

            for particle in particles:
                particle.update_velocity(inertia_weight, alpha, beta, global_best)
                particle.update_position(position_limits)
            
            curr_iter += 1
        
        computation_list.append(time.time()-start)
        print(time.time()-start)
    
    plt.figure()
    plt.plot([x*10 for x in range(1,101)], computation_list)
    plt.ylabel("Computational Speed")
    plt.xlabel("Population")
    plt.title("Computational time vs Population")
    plt.savefig("Computational_time_vs_Population",bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()