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

    ### using cost and levels to calculate fitness

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

def calc_avg_pos_diff(particles):
    position_list = [particle.position for particle in particles]
    mean_poition = sum(position_list) / len(position_list)
    difference = [abs(pos - mean_poition) for pos in position_list]
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
    run_vals = [i for i in range(1,101)]
    frequent_particle_list = []
    global_best_list = []
    iteration_list = []
    final_position_list = []
    time_complexity_list = []
    space_ax = plt.axes()

    for run in run_vals:
        random.seed(run)
        # fixed variables
        position_cap = 10079 # minute representation of Sunday 11:59
        position_limits = [0, position_cap]

        # termination condition
        max_iter = 100
        min_avg_fit_diff = 0.01
        min_avg_dis_diff = 0.01

        # parameters 
        alpha = [0.5, 0.5]
        alpha_change = 0.1 # How much alpha values increase or decrease at split
        inertia_weight = 1
        inertia_change = 0.2 # How much inertia weight decrease at split
        iteration_split = 5 # Iteration split - alpha and inertia values increase/decrease at split
        num_particles = 100
        spawn_segments = 10 # Initialization splits
        
        # initialization
        global_best = None
        curr_iter = 0

        start_time = time.time()
        particles = initialize_particles(num_particles, position_limits, spawn_segments)

        if len(space_ax.lines) > 1:
            del space_ax.lines[1]
        space_ax.plot(list(range(*position_limits)),[calc_fitness(x) for x in range(*position_limits)])
        space_ax.set_title("Run: {} Iteration: {}".format(run, curr_iter))
        space_ax.set_xlabel("Position")
        space_ax.set_ylabel("Fitness")

        global_best = particles[0].position
        while((curr_iter < max_iter) and (calc_avg_fit_diff(particles) > min_avg_fit_diff) and (calc_avg_pos_diff(particles) > min_avg_dis_diff)):

            if len(space_ax.lines) > 1:
                del space_ax.lines[1]
            space_ax.plot([x.position for x in particles], [calc_fitness(x.position) for x in particles], 'go')
            space_ax.set_title("Run: {} Iteration: {}".format(run, curr_iter))
            plt.pause(0.01) 

            if (curr_iter % (max_iter//iteration_split) == 0) and (curr_iter != 0):
                inertia_weight -= inertia_change
                alpha[0] -= alpha_change
                alpha[1] += alpha_change

            for particle in particles:
                particle.update_local_best()
                global_best = check_global_best(particle.local_best, global_best)
         
            beta = [random.random(), random.random()]

            for particle in particles:
                particle.update_velocity(inertia_weight, alpha, beta, global_best)
                particle.update_position(position_limits)
            
            curr_iter += 1

        particle_position = [particle.position for particle in particles]
        most_frequent_particle = max(set(particle_position), key = particle_position.count)
        print("Final Results of Run {}:".format(run))
        print("Iteration:", curr_iter)
        print("Most Frequent Particles:", most_frequent_particle, str(particle_position.count(most_frequent_particle))+"%", "Fitness:", calc_fitness(most_frequent_particle))
        print("Global best:", global_best, "Fitness:",  calc_fitness(global_best))
        print("Cost:", (calc_acc_cost(global_best) + calc_mov_cost(global_best)), "Renovation:", calc_ren_lvl(global_best))
        
        if len(space_ax.lines) > 1:
            del space_ax.lines[1]
        space_ax.plot([x.position for x in particles], [calc_fitness(x.position) for x in particles], 'go')
        space_ax.set_title("Run: {} Completed, Iteration taken: {}".format(run, curr_iter))
        plt.pause(0.5)

        time_complexity_list.append(time.time()-start_time)
        final_position_list.append(particle_position.copy())
        iteration_list.append(curr_iter)
        global_best_list.append(global_best)
        frequent_particle_list.append(most_frequent_particle)
    plt.close()

    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(2,1,1, xlabel="Runs", ylabel="Iterations", title="Iteration of each Run")
    ax1.plot(run_vals, iteration_list)
    ax2 = fig.add_subplot(2,1,2, xlabel="Runs", ylabel="Time Taken", title="Time taken for each Run")
    ax2.plot(run_vals, time_complexity_list)
    plt.savefig("Iterations and Time complexity",bbox_inches='tight')
    plt.pause(2)
    plt.close()

    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(2,1,1, xlabel="Runs", ylabel="Global best position", title="Global best position of each Run")
    ax1.plot(run_vals, global_best_list)
    ax2 = fig.add_subplot(2,1,2, xlabel="Runs", ylabel="Most frequent Position", title="Most frequent position of each Run")
    ax2.plot(run_vals, frequent_particle_list)
    plt.savefig("Global best position and Most frequent position",bbox_inches='tight')
    plt.pause(2)
    plt.close()

    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(2,1,1, xlabel="Runs", ylabel="Optimal cost", title="Optimal cost of each Run")
    ax1.plot(run_vals, [x + y for x, y in zip(list(map(calc_mov_cost, frequent_particle_list)), list(map(calc_acc_cost,frequent_particle_list)))])
    ax1.plot(run_vals, [x + y for x, y in zip(list(map(calc_mov_cost, global_best_list)), list(map(calc_acc_cost,global_best_list)))], 'go--')
    ax2 = fig.add_subplot(2,1,2, xlabel="Runs", ylabel="Optimal renovation level", title="Optimal renovation level of each Run")
    ax2.plot(run_vals, list(map(calc_ren_lvl, frequent_particle_list)))
    ax2.plot(run_vals, list(map(calc_ren_lvl, global_best_list)), 'go--')
    plt.savefig("Optimal Cost and Renovation Level",bbox_inches='tight')
    plt.pause(2)
    plt.close()

    plt.figure()
    for run in run_vals:
        plt.scatter(final_position_list[run-1], [run for i in range(len(final_position_list[run-1]))])
    plt.title("Particle's final position of each Run")
    plt.ylabel("Run")
    plt.xlabel("Position value")
    plt.savefig("Final particle positions",bbox_inches='tight')
    plt.pause(2)
    plt.close()
    
if __name__ == "__main__":
    main()