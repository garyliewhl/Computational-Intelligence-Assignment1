import numpy as np
import random
import math
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, position, velocity = 0):
        self.position = position
        self.velocity = velocity
        self.local_best = position
        self.position_list = [position]
        self.velocity_list = [velocity]
        self.best_position_list = []
    
    def update_position(self, position_limits):
        self.position = round((self.position + self.velocity), 0)

        # set position to the min or max value if it goes out of bounds
        if self.position > position_limits[1]:
            self.position = position_limits[1]
        elif self.position < position_limits[0]:
            self.position = position_limits[0]
        self.position_list.append(self.position)
    
    def update_velocity(self, inertia, alpha, beta, global_best):
        self.velocity = (inertia * self.velocity) + (alpha[0] * beta[0] * (self.local_best - self.position)) + (alpha[1] * beta[1] * (global_best - self.position))
        self.velocity_list.append(self.velocity)

    def update_local_best(self):
        if calc_fitness(self.position) > calc_fitness(self.local_best):
            self.local_best = self.position
        self.best_position_list.append(self.local_best)

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
    fitness = renovation_level
    fitness += 1 - (((accommodation_cost - acc_cost_range[0]) + (moving_cost - mov_cost_range[0])) / ((acc_cost_range[1] - acc_cost_range[0]) + (mov_cost_range[1] - mov_cost_range[0])))

    ### fitness function 3, slight modification of fitness function 3.
    # fitness = renovation_level * ((mov_cost_range[1] - mov_cost_range[0]) + (acc_cost_range[1] - acc_cost_range[0]))
    # fitness -= (accommodation_cost + moving_cost)

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
    position_list = [particle.position for particle in particles]
    mean_poition = sum(position_list) / len(position_list)
    difference = [abs(pos - mean_poition) for pos in position_list]
    avg_pos_diff = sum(difference)/len(difference)
    return avg_pos_diff

def initialize_particles(num_particles, position_limits):
    particles = []
    segments = 10
    pop_segment = num_particles//segments
    range_val = position_limits[1]//segments
    min_val = 0
    max_val=range_val
    for segment in range(segments):
        for num in range(pop_segment):
            particles.append(Particle(random.randint(position_limits[0], position_limits[1])))
        min_val+=range_val
        max_val+=range_val
    return particles

# def initialize_particles(num_particles, position_limits):
#     particles = []
#     for num in range(num_particles):
#         particles.append(Particle(random.randint(position_limits[0], position_limits[1])))
#     return particles

def main():
    # fixed variables
    position_cap = 10079 # minute representation of Sunday 11:59

    # parameters initialization
    alpha = [0.5, 0.5]
    inertia_weight = 0.6
    num_particles = 70
    global_best = None
    position_limits = [0, position_cap]
    global_best_position_list = []
    curr_iter = 0

    # termination condition
    max_iter = 200
    min_avg_fit_diff = 0.01
    min_avg_dis_diff = 0.01

    particles = initialize_particles(num_particles, position_limits)

    space_ax = plt.axes()
    space_ax.plot(list(range(*position_limits)),[calc_fitness(x) for x in range(*position_limits)])
    space_ax.set_title("Position of particles in iteration {}".format(curr_iter))
    space_ax.set_xlabel("Position")
    space_ax.set_ylabel("Fitness")

    global_best = particles[0].position
    while((curr_iter < max_iter) and (calc_avg_fit_diff(particles) > min_avg_fit_diff) and (calc_avg_pos_diff(particles) > min_avg_dis_diff)):

        if len(space_ax.lines) > 1:
            del space_ax.lines[1]
        space_ax.plot([x.position for x in particles], [calc_fitness(x.position) for x in particles], 'go')
        space_ax.set_title("Position of particles in iteration {}".format(curr_iter))
        plt.pause(0.05) 

        if curr_iter % (max_iter//5) == 0:
            inertia_weight -= 0.1
            alpha[0] -= 0.1
            alpha[1] += 0.1

        for particle in particles:
            particle.update_local_best()
            global_best = check_global_best(particle.local_best, global_best)

        global_best_position_list.append(global_best)           
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
    
    if len(space_ax.lines) > 1:
        del space_ax.lines[1]
    space_ax.plot([x.position for x in particles], [calc_fitness(x.position) for x in particles], 'go')
    space_ax.set_title("Position of particles in iteration {}".format(curr_iter))
    
    [pos_fig, position_axes] = plt.subplots(4,1,sharex=True)
    position_axes[0].set_title("Position of each particle")
    position_axes[1].set_title("Fitness of each particle")
    position_axes[2].set_title("Boxplot of position at each iteration")
    position_axes[3].set_title("Boxplot of fitness at each iteration")
    position_axes[3].set_xlabel("Iteration")
    [vel_fig, velocity_axes] = plt.subplots(2,1,sharex=True)
    velocity_axes[0].set_title("Velocity of each particle")
    velocity_axes[1].set_title("Boxplot for velocity at each iteration")
    velocity_axes[1].set_xlabel("Iteration")
    [p_best_fig, personal_best_axes] = plt.subplots(4,1,sharex=True)
    personal_best_axes[0].set_title("Personal best position of each particle")
    personal_best_axes[1].set_title("Personal best fitness of each particle")
    personal_best_axes[2].set_title("Boxplot of personal best position at each iteration")
    personal_best_axes[3].set_title("Boxplot of personal best fitness at each iteration")
    personal_best_axes[3].set_xlabel("Iteration")
    [g_best_fig, global_best_axes] = plt.subplots(2,1,sharex=True)
    global_best_axes[0].set_title("Global best position")
    global_best_axes[1].set_title("Boxplot for global best position")
    global_best_axes[1].set_xlabel("Iteration")
    
    for particle in particles:
        iteration_list = list(range(len(particle.position_list)))
        position_axes[0].plot(iteration_list, particle.position_list, '-o')
        position_axes[1].plot(iteration_list, [calc_fitness(x) for x in particle.position_list], '-o')

        velocity_axes[0].plot(iteration_list, particle.velocity_list, '-o')

        personal_best_axes[0].plot(iteration_list[:-1], particle.best_position_list, '-o')
        personal_best_axes[1].plot(iteration_list[:-1], [calc_fitness(x) for x in particle.best_position_list], '-o')

    position_axes[2].boxplot([[p.position_list[i] for p in particles] for i in iteration_list], positions=iteration_list)
    position_axes[3].boxplot([[calc_fitness(p.position_list[i]) for p in particles] for i in iteration_list], positions=iteration_list)

    velocity_axes[1].boxplot([[p.velocity_list[i] for p in particles] for i in iteration_list], positions=iteration_list)

    personal_best_axes[2].boxplot([[p.best_position_list[i] for p in particles] for i in iteration_list[:-1]], positions=iteration_list[:-1])
    personal_best_axes[3].boxplot([[calc_fitness(p.best_position_list[i]) for p in particles] for i in iteration_list[:-1]], positions=iteration_list[:-1])

    global_best_axes[0].plot(iteration_list[:-1], global_best_position_list, '-o')
    global_best_axes[1].plot(iteration_list[:-1], [calc_fitness(x) for x in global_best_position_list], '-o')
    plt.show()
    
if __name__ == "__main__":
    main()